"""Polymarket / Polygon client — the *only* file in the project that touches the
Polymarket CLOB API, the Polymarket Gamma API, or the Polygon chain directly.

All other modules (orchestrator, decide, store, ...) call PolymarketClient methods.

API surface:
    PolymarketClient(...)
        .get_usdc_balance() -> float
        .get_matic_balance() -> float
        .find_btc_15m_market(window_start_unix_s) -> dict
        .place_market_order(token_id, usdc_amount) -> dict
        .read_settlement(condition_id) -> dict | None

Auth: EOA (POLYGON_PRIVATE_KEY signs orders) + Polymarket proxy (POLYMARKET_PROXY_ADDRESS
holds the USDC). API creds are derived from the private key via py_clob_client at boot
and cached in memory.
"""
from __future__ import annotations

import logging
import time

import requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import MarketOrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY
from web3 import Web3

import settings

logger = logging.getLogger("live.polymarket")

# USDC.e (native USDC on Polygon, the one Polymarket uses).
USDC_POLYGON_ADDRESS = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
USDC_DECIMALS = 6

# Minimal ERC-20 ABI: only balanceOf is needed.
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    }
]

CHAIN_ID_POLYGON = 137
SIGNATURE_TYPE_PROXY = 2  # py-clob-client constant for "Polymarket proxy wallet"


class MarketNotFoundError(Exception):
    """Raised when find_btc_15m_market doesn't find a market for the given window."""


def _parse_amount(s) -> float:
    """Polymarket CLOB returns takingAmount/makingAmount as strings (or "" when unfilled).
    Defensive cast: empty/None -> 0.0; otherwise float(str)."""
    if s is None or s == "":
        return 0.0
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _failed_order(error_msg: str) -> dict:
    """Normalized failure response for orchestrator consumption."""
    return {
        "order_id": None,
        "status": "failed",
        "average_fill_price": None,
        "shares_filled": 0.0,
        "usdc_spent": 0.0,
        "fee_usdc": None,
        "error_msg": error_msg,
        "raw": {"error": error_msg},
    }


class PolymarketClient:
    def __init__(
        self,
        private_key: str,
        eoa_address: str,
        proxy_address: str,
        rpc_url: str,
        clob_host: str,
        gamma_host: str,
        timeout: float = settings.POLYMARKET_API_TIMEOUT,
    ):
        self.eoa_address = Web3.to_checksum_address(eoa_address)
        self.proxy_address = Web3.to_checksum_address(proxy_address)
        self.gamma_host = gamma_host.rstrip("/")
        self.timeout = timeout

        # web3 for direct Polygon reads (USDC.balanceOf, MATIC balance).
        self.web3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": timeout}))
        self.usdc = self.web3.eth.contract(address=USDC_POLYGON_ADDRESS, abi=ERC20_ABI)

        # CLOB client (signs + places orders).
        self.clob = ClobClient(
            host=clob_host,
            key=private_key,
            chain_id=CHAIN_ID_POLYGON,
            signature_type=SIGNATURE_TYPE_PROXY,
            funder=self.proxy_address,
        )
        self.clob.set_api_creds(self.clob.create_or_derive_api_creds())

    # ---------------- Balances ----------------

    def get_usdc_balance(self) -> float:
        """Free USDC on the Polymarket proxy wallet, in human units."""
        wei = self.usdc.functions.balanceOf(self.proxy_address).call()
        return wei / (10 ** USDC_DECIMALS)

    def get_matic_balance(self) -> float:
        """Native MATIC balance on the EOA (used for gas)."""
        wei = self.web3.eth.get_balance(self.eoa_address)
        return wei / (10 ** 18)

    # ---------------- Market discovery ----------------

    def find_btc_15m_market(self, window_start_unix_s: int) -> dict:
        """Look up the market for the 15min window starting at window_start_unix_s.

        Slug pattern observed from polymarket.com/event/btc-updown-15m-<unix_s>.

        Returns:
            {condition_id, up_token_id, down_token_id, market_slug, end_date_iso, status}

        Raises:
            MarketNotFoundError on 404 / empty response.
        """
        slug = f"btc-updown-15m-{window_start_unix_s}"
        url = f"{self.gamma_host}/events"

        last_err: Exception | None = None
        for attempt in range(2):  # 1 retry on transient
            try:
                r = requests.get(url, params={"slug": slug}, timeout=self.timeout)
                r.raise_for_status()
                events = r.json()
                if not events:
                    raise MarketNotFoundError(f"Slug not found: {slug}")
                event = events[0] if isinstance(events, list) else events
                markets = event.get("markets") or []
                if not markets:
                    raise MarketNotFoundError(f"Slug {slug} has no markets")
                market = markets[0]
                # Polymarket binary markets carry two clobTokenIds: [Yes-of-outcome-0, Yes-of-outcome-1].
                # Outcomes are typically ['Up', 'Down'] (verify at smoke-test time).
                outcomes = market.get("outcomes") or '["Up", "Down"]'
                if isinstance(outcomes, str):
                    import json
                    outcomes = json.loads(outcomes)
                token_ids = market.get("clobTokenIds") or "[]"
                if isinstance(token_ids, str):
                    import json
                    token_ids = json.loads(token_ids)

                up_idx = next((i for i, o in enumerate(outcomes) if str(o).lower() == "up"), 0)
                down_idx = 1 - up_idx
                return {
                    "condition_id": market["conditionId"],
                    "up_token_id": token_ids[up_idx],
                    "down_token_id": token_ids[down_idx],
                    "market_slug": slug,
                    "end_date_iso": market.get("endDate"),
                    "status": "open" if not market.get("closed") else "closed",
                }
            except MarketNotFoundError:
                raise
            except (requests.exceptions.RequestException, ValueError) as e:
                last_err = e
                if attempt == 0:
                    time.sleep(1)
                    continue
                raise
        raise RuntimeError(f"unreachable: last_err={last_err}")  # pragma: no cover

    # ---------------- Order placement ----------------

    def place_market_order(self, token_id: str, usdc_amount: float) -> dict:
        """FOK market BUY of `usdc_amount` USDC worth of `token_id` shares.

        Polymarket CLOB POST /order response shape (per docs):
            {
              "success": bool,
              "errorMsg": str,
              "orderID": str,
              "status": "live" | "matched" | "delayed" | "unmatched",
              "takingAmount": str,    # shares received (taker side); "" if unfilled
              "makingAmount": str,    # USDC paid (maker side); "" if unfilled
              "transactionsHashes": list[str],
              "tradeIDs": list[str],
            }
        For a FOK BUY: success=True + status="matched" if filled,
                       success=False + errorMsg="FOK_ORDER_NOT_FILLED_ERROR" if depth insufficient.

        Returns (our normalized shape):
            {order_id, status, average_fill_price, shares_filled, usdc_spent,
             fee_usdc, error_msg, raw}
            status in {"filled", "rejected", "failed"}
            fee_usdc is None for now — Polymarket fees aren't echoed in the order response.
        """
        for attempt in range(2):
            try:
                args = MarketOrderArgs(
                    token_id=token_id,
                    amount=float(usdc_amount),  # USDC for BUY market orders
                    side=BUY,
                    order_type=OrderType.FOK,
                )
                signed = self.clob.create_market_order(args)
                resp = self.clob.post_order(signed, OrderType.FOK)

                success = bool(resp.get("success", False))
                error_msg = resp.get("errorMsg") or ""
                order_id = resp.get("orderID")
                api_status = resp.get("status")  # "live" | "matched" | ...

                if not success or api_status != "matched":
                    return {
                        "order_id": order_id,
                        "status": "rejected",
                        "average_fill_price": None,
                        "shares_filled": 0.0,
                        "usdc_spent": 0.0,
                        "fee_usdc": None,
                        "error_msg": error_msg or f"unexpected status={api_status}",
                        "raw": resp,
                    }

                # Filled. takingAmount/makingAmount are JSON-decoded as strings; convert defensively.
                shares_filled = _parse_amount(resp.get("takingAmount"))
                usdc_spent = _parse_amount(resp.get("makingAmount"))
                avg_price = (usdc_spent / shares_filled) if shares_filled > 0 else None

                return {
                    "order_id": order_id,
                    "status": "filled",
                    "average_fill_price": avg_price,
                    "shares_filled": shares_filled,
                    "usdc_spent": usdc_spent,
                    "fee_usdc": None,
                    "error_msg": "",
                    "raw": resp,
                }
            except requests.exceptions.RequestException as e:
                if attempt == 0:
                    logger.warning("place_market_order network failure (attempt 1): %s", e)
                    time.sleep(1)
                    continue
                logger.error("place_market_order network failed twice: %s", e)
                return _failed_order(str(e))
            except Exception as e:
                logger.exception("place_market_order unexpected error: %s", e)
                return _failed_order(str(e))

    # ---------------- Settlement read ----------------

    def read_settlement(self, market_slug: str) -> dict | None:
        """After window close, query Gamma for resolution via the same slug used at discovery.

        Verified empirically (2026-05-08) :
            - GET /events?slug=<slug> returns the resolved market with closed=True and
              outcomePrices=["1","0"] or ["0","1"] once settled.
            - GET /markets?condition_id=<...> silently ignores the filter and returns
              an unfiltered list — do NOT use it.
            - GET /markets/<numeric_id> works but requires storing the numeric id, which
              we don't keep.

        Returns {resolved: bool, winning_outcome: 'up'|'down', resolved_at: iso}
        if settled; None if still pending or on transient error.
        """
        url = f"{self.gamma_host}/events"
        try:
            r = requests.get(url, params={"slug": market_slug}, timeout=self.timeout)
            r.raise_for_status()
            events = r.json()
            if not events:
                return None
            event = events[0] if isinstance(events, list) else events
            markets = event.get("markets") or []
            if not markets:
                return None
            market = markets[0]
            if not market.get("closed"):
                return None
            outcomes = market.get("outcomes") or '["Up","Down"]'
            outcome_prices = market.get("outcomePrices") or '["0","0"]'
            if isinstance(outcomes, str):
                import json
                outcomes = json.loads(outcomes)
            if isinstance(outcome_prices, str):
                import json
                outcome_prices = json.loads(outcome_prices)
            # Winner = outcome whose settled price == "1".
            winner_idx = next(
                (i for i, p in enumerate(outcome_prices) if str(p) in ("1", "1.0")),
                None,
            )
            if winner_idx is None:
                return None
            winning = "up" if str(outcomes[winner_idx]).lower() == "up" else "down"
            return {
                "resolved": True,
                "winning_outcome": winning,
                "resolved_at": market.get("endDate"),
            }
        except (requests.exceptions.RequestException, ValueError) as e:
            logger.warning("read_settlement failed for slug=%s: %s", market_slug, e)
            return None


def from_settings() -> PolymarketClient:
    """Convenience constructor reading from settings module."""
    settings.assert_env_ready()
    return PolymarketClient(
        private_key=settings.POLYGON_PRIVATE_KEY,
        eoa_address=settings.POLYGON_EOA_ADDRESS,
        proxy_address=settings.POLYMARKET_PROXY_ADDRESS,
        rpc_url=settings.POLYGON_RPC_URL,
        clob_host=settings.CLOB_HOST,
        gamma_host=settings.GAMMA_HOST,
    )


if __name__ == "__main__":
    # Smoke test: instantiate, read balances, look up the next 15min market.
    import time
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = from_settings()
    print(f"USDC balance : {client.get_usdc_balance():.4f}")
    print(f"MATIC balance: {client.get_matic_balance():.6f}")
    next_boundary_s = ((int(time.time()) // 900) + 1) * 900
    try:
        market = client.find_btc_15m_market(next_boundary_s)
        print(f"Next market : {market}")
    except MarketNotFoundError as e:
        print(f"No market for next boundary {next_boundary_s}: {e}")
