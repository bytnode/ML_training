"""Polymarket / Polygon client - the *only* file in the project that touches
the Polymarket CLOB, the Polymarket Gamma API, or the Polygon chain directly.

All other modules (live.py, notify.py, live_helpers/*) call PolymarketClient
methods. The day the API changes, this is the single file to audit.

API surface:
    PolymarketClient.from_env()  -> read env vars and build a client
        .get_usdc_balance()      -> float
        .get_matic_balance()     -> float
        .find_btc_15m_market(window_start_unix_s) -> dict
        .place_fak_order(token_id, usdc_amount)  -> dict
        .read_settlement(slug)   -> dict | None
        .redeem(condition_id)    -> str | None   (tx hash)

Auth: EOA Polygon (POLYGON_PRIVATE_KEY signs orders) + Polymarket proxy
(POLYMARKET_PROXY_ADDRESS holds the USDC). API creds derived from the private
key via py-clob-client at boot, kept in memory only.

See docs/superpowers/specs/2026-05-14-polymarket-live-betting-design.md
for the full design + the empirical findings that drove these choices.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any

import requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY
from web3 import Web3

logger = logging.getLogger("live.polymarket")

# --- Polygon constants ---

# USDC.e (the native USDC on Polygon, 6 decimals) - what the proxy wallet holds.
USDC_POLYGON_ADDRESS = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
USDC_DECIMALS = 6

# pUSD - Polymarket's 1:1 wrapper of USDC.e used as CTF collateral. redeemPositions
# returns pUSD; the proxy Safe should auto-unwrap to USDC.e (to verify at smoke test).
PUSD_POLYGON_ADDRESS = Web3.to_checksum_address("0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB")

# ConditionalTokens framework on Polygon - target of redeemPositions().
CTF_CONTRACT_ADDRESS = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")

CHAIN_ID_POLYGON = 137
SIGNATURE_TYPE_PROXY = 2   # py-clob-client constant: "Polymarket proxy wallet"

# Min order size imposed by Polymarket (in shares). Below this the CLOB rejects.
POLYMARKET_MIN_SHARES = 5
# Tick size for binary markets. Limit prices must round to this.
TICK_SIZE = 0.01
# Hard upper bound on the slippage-capped limit (a binary YES never trades above 0.99).
MAX_LIMIT_PRICE = 0.99

DEFAULT_CLOB_HOST = "https://clob.polymarket.com"
DEFAULT_GAMMA_HOST = "https://gamma-api.polymarket.com"

# --- ABIs (minimal subsets, only what we call) ---

ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    }
]

CTF_ABI = [
    {
        "inputs": [
            {"internalType": "address",   "name": "collateralToken",    "type": "address"},
            {"internalType": "bytes32",   "name": "parentCollectionId", "type": "bytes32"},
            {"internalType": "bytes32",   "name": "conditionId",        "type": "bytes32"},
            {"internalType": "uint256[]", "name": "indexSets",          "type": "uint256[]"},
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]


class MarketNotFoundError(Exception):
    """Raised when find_btc_15m_market doesn't find a market for the given window."""


def _parse_amount(s: Any) -> float:
    """Polymarket CLOB returns takingAmount/makingAmount as strings (or "" when
    unfilled). Defensive cast: empty/None -> 0.0; bad input -> 0.0."""
    if s is None or s == "":
        return 0.0
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _failed_order(error_msg: str, raw: dict | None = None) -> dict:
    """Normalized failure shape so live.py never has to second-guess the dict keys."""
    return {
        "order_id": None,
        "status": "failed",
        "fill_price_avg": None,
        "shares_filled": 0.0,
        "usdc_spent": 0.0,
        "fee_usdc": None,
        "error_msg": error_msg,
        "raw": raw or {"error": error_msg},
    }


def _rejected_order(error_msg: str, order_id: str | None, raw: dict) -> dict:
    return {
        "order_id": order_id,
        "status": "rejected",
        "fill_price_avg": None,
        "shares_filled": 0.0,
        "usdc_spent": 0.0,
        "fee_usdc": None,
        "error_msg": error_msg,
        "raw": raw,
    }


class PolymarketClient:
    def __init__(
        self,
        private_key: str,
        eoa_address: str,
        proxy_address: str,
        rpc_url: str,
        clob_host: str = DEFAULT_CLOB_HOST,
        gamma_host: str = DEFAULT_GAMMA_HOST,
        slippage_bps: int = 200,
        timeout: float = 10.0,
    ):
        self.eoa_address = Web3.to_checksum_address(eoa_address)
        self.proxy_address = Web3.to_checksum_address(proxy_address)
        self.gamma_host = gamma_host.rstrip("/")
        self.timeout = timeout
        self.slippage_bps = slippage_bps

        # web3 client for direct Polygon reads + the redeemPositions tx.
        self.web3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": timeout}))
        self.usdc = self.web3.eth.contract(address=USDC_POLYGON_ADDRESS, abi=ERC20_ABI)
        self.ctf = self.web3.eth.contract(address=CTF_CONTRACT_ADDRESS, abi=CTF_ABI)
        self._private_key = private_key

        # CLOB client. signature_type=2 = the funder is a Polymarket proxy (Gnosis Safe)
        # holding the USDC; the EOA we sign with is the controller of that Safe.
        self.clob = ClobClient(
            host=clob_host,
            key=private_key,
            chain_id=CHAIN_ID_POLYGON,
            signature_type=SIGNATURE_TYPE_PROXY,
            funder=self.proxy_address,
        )
        # L2 API credentials are derived from the private key. Kept in memory only.
        self.clob.set_api_creds(self.clob.create_or_derive_api_creds())

    @classmethod
    def from_env(cls, slippage_bps: int | None = None) -> "PolymarketClient":
        """Build a client from .env. Raises RuntimeError listing every missing var."""
        required = [
            "POLYGON_PRIVATE_KEY",
            "POLYGON_EOA_ADDRESS",
            "POLYMARKET_PROXY_ADDRESS",
            "POLYGON_RPC_URL",
        ]
        missing = [k for k in required if not os.environ.get(k)]
        if missing:
            raise RuntimeError(
                f"PolymarketClient.from_env: missing env vars {missing}. "
                f"Set them in live/.env or unset LIVE_MODE to fall back to paper mode."
            )
        if slippage_bps is None:
            slippage_bps = int(os.environ.get("SLIPPAGE_BPS", "200"))
        return cls(
            private_key=os.environ["POLYGON_PRIVATE_KEY"],
            eoa_address=os.environ["POLYGON_EOA_ADDRESS"],
            proxy_address=os.environ["POLYMARKET_PROXY_ADDRESS"],
            rpc_url=os.environ["POLYGON_RPC_URL"],
            clob_host=os.environ.get("CLOB_HOST", DEFAULT_CLOB_HOST),
            gamma_host=os.environ.get("GAMMA_HOST", DEFAULT_GAMMA_HOST),
            slippage_bps=slippage_bps,
        )

    # ============================================================
    #  Balances (direct Polygon RPC reads, no Polymarket API call)
    # ============================================================

    def get_usdc_balance(self) -> float:
        """Free USDC.e on the proxy wallet, in human units."""
        wei = self.usdc.functions.balanceOf(self.proxy_address).call()
        return wei / (10 ** USDC_DECIMALS)

    def get_matic_balance(self) -> float:
        """Native MATIC on the EOA (gas for redeemPositions)."""
        return self.web3.eth.get_balance(self.eoa_address) / (10 ** 18)

    # ============================================================
    #  Market discovery (Gamma)
    # ============================================================

    def find_btc_15m_market(self, window_start_unix_s: int) -> dict:
        """Look up the BTC up/down 15min market whose window starts at
        `window_start_unix_s` (UTC seconds, multiple of 900).

        Returns:
            {
              "condition_id":  hex string,
              "up_token_id":   ERC-1155 token id (str),
              "down_token_id": ERC-1155 token id (str),
              "market_slug":   "btc-updown-15m-<unix_s>",
              "end_date_iso":  ISO timestamp,
              "status":        "open" | "closed",
            }

        Raises MarketNotFoundError on 404 or empty response.
        Retries once on transient HTTP errors before raising.
        """
        slug = f"btc-updown-15m-{window_start_unix_s}"
        url = f"{self.gamma_host}/markets/slug/{slug}"

        last_err: Exception | None = None
        for attempt in range(2):
            try:
                r = requests.get(url, timeout=self.timeout)
                if r.status_code == 404:
                    raise MarketNotFoundError(f"Slug not found: {slug}")
                r.raise_for_status()
                market = r.json()
                # Gamma returns a single market dict on /markets/slug/<slug>.
                if not market:
                    raise MarketNotFoundError(f"Empty response for slug: {slug}")
                return self._parse_market(market, slug)
            except MarketNotFoundError:
                raise
            except (requests.exceptions.RequestException, ValueError) as e:
                last_err = e
                if attempt == 0:
                    logger.warning("find_btc_15m_market transient error, retrying: %s", e)
                    time.sleep(1)
                    continue
                raise
        # unreachable - the loop either returns or raises
        raise RuntimeError(f"find_btc_15m_market: unreachable, last_err={last_err}")

    @staticmethod
    def _parse_market(market: dict, slug: str) -> dict:
        """Decode the Gamma market payload into our standard shape.

        outcomes / clobTokenIds come back as JSON-encoded strings (per the
        research notes on Polymarket payload conventions). Always json.loads()
        and never trust positional order - look at `outcomes` to find which
        index is "Up" vs "Down".
        """
        outcomes = market.get("outcomes", '["Up", "Down"]')
        token_ids = market.get("clobTokenIds", "[]")
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        if not isinstance(outcomes, list) or not isinstance(token_ids, list):
            raise RuntimeError(f"Unexpected outcomes/tokens shape: {outcomes=} {token_ids=}")
        if len(outcomes) != 2 or len(token_ids) != 2:
            raise RuntimeError(f"Expected 2 outcomes, got {len(outcomes)}: {outcomes}")

        up_idx = next((i for i, o in enumerate(outcomes) if str(o).lower() == "up"), None)
        if up_idx is None:
            raise RuntimeError(f"No 'Up' outcome in {outcomes}")
        down_idx = 1 - up_idx
        return {
            "condition_id":  market["conditionId"],
            "up_token_id":   str(token_ids[up_idx]),
            "down_token_id": str(token_ids[down_idx]),
            "market_slug":   slug,
            "end_date_iso":  market.get("endDate"),
            "status":        "closed" if market.get("closed") else "open",
        }

    # ============================================================
    #  Order placement
    # ============================================================

    def place_fak_order(self, token_id: str, usdc_amount: float) -> dict:
        """Place a Fill-And-Kill BUY order on `token_id` for ~`usdc_amount` USDC.

        Flow:
        1. Read best ask from /book?token_id=...
        2. limit_price = min(round(best_ask + slippage_bps/10000, 2), 0.99)
        3. size_shares = floor(usdc_amount / limit_price)   (Polymarket trades whole shares)
        4. If size_shares < 5 -> reject (below Polymarket's 5-share minimum)
        5. Sign + POST an OrderArgs at (price=limit_price, size=size_shares, side=BUY)
           with OrderType.FAK.

        FAK semantics: fill what the book offers at price <= limit, kill the rest.
        No leftover resting order. status="filled" even on partial fill (we record
        the actual shares_filled / usdc_spent).

        Returns:
            {
              order_id, status, fill_price_avg, shares_filled,
              usdc_spent, fee_usdc, error_msg, raw,
            }
            status in {"filled", "rejected", "failed"}.
        """
        # 1. Read book.
        try:
            best_ask = self._read_best_ask(token_id)
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            return _failed_order(f"book read failed: {e}")

        # 2. Slippage-capped limit, rounded to tick.
        raw_limit = best_ask + self.slippage_bps / 10000
        limit_price = min(round(raw_limit, 2), MAX_LIMIT_PRICE)
        if limit_price <= 0 or limit_price >= 1:
            return _failed_order(f"limit_price out of range: {limit_price} (best_ask={best_ask})")

        # 3. Size in shares (whole shares only).
        size_shares = math.floor(usdc_amount / limit_price)
        if size_shares < POLYMARKET_MIN_SHARES:
            return _rejected_order(
                f"size={size_shares} below {POLYMARKET_MIN_SHARES}-share min "
                f"(usdc={usdc_amount}, limit={limit_price})",
                order_id=None,
                raw={"reason": "size_below_min"},
            )

        # 4 + 5. Sign + post FAK.
        for attempt in range(2):
            try:
                args = OrderArgs(
                    token_id=token_id,
                    price=float(limit_price),
                    size=float(size_shares),
                    side=BUY,
                )
                signed = self.clob.create_order(args)
                resp = self.clob.post_order(signed, OrderType.FAK)
                return self._interpret_order_response(resp)
            except requests.exceptions.RequestException as e:
                if attempt == 0:
                    logger.warning("post_order network failure (attempt 1): %s", e)
                    time.sleep(1)
                    continue
                logger.error("post_order network failed twice: %s", e)
                return _failed_order(str(e))
            except Exception as e:
                logger.exception("post_order unexpected error: %s", e)
                return _failed_order(str(e))
        # unreachable
        return _failed_order("post_order loop exited without return")

    def _read_best_ask(self, token_id: str) -> float:
        """Hit the CLOB book endpoint and return the lowest ask price (decimal)."""
        url = f"{self.clob.host.rstrip('/')}/book"
        r = requests.get(url, params={"token_id": token_id}, timeout=self.timeout)
        r.raise_for_status()
        book = r.json()
        asks = book.get("asks") or []
        if not asks:
            raise ValueError(f"no asks on the book for token_id={token_id}")
        # Polymarket book asks are sorted ascending by price (verified empirically).
        return float(asks[0]["price"])

    @staticmethod
    def _interpret_order_response(resp: dict) -> dict:
        """Map the raw POST /order response into our normalized shape.

        Per Polymarket docs:
            {
              "success":    bool,
              "errorMsg":   str,
              "orderID":    str,
              "status":     "live" | "matched" | "delayed" | "unmatched",
              "takingAmount": str,   # shares received (taker side); "" if unfilled
              "makingAmount": str,   # USDC paid (maker side); "" if unfilled
              "transactionsHashes": list[str],
              "tradeIDs":   list[str],
            }
        """
        success = bool(resp.get("success", False))
        error_msg = resp.get("errorMsg") or ""
        order_id = resp.get("orderID")
        api_status = resp.get("status")

        shares_filled = _parse_amount(resp.get("takingAmount"))
        usdc_spent = _parse_amount(resp.get("makingAmount"))

        # Filled (full or partial) iff CLOB matched any shares at our limit.
        if success and shares_filled > 0:
            avg_price = (usdc_spent / shares_filled) if shares_filled > 0 else None
            return {
                "order_id": order_id,
                "status": "filled",
                "fill_price_avg": avg_price,
                "shares_filled": shares_filled,
                "usdc_spent": usdc_spent,
                "fee_usdc": None,  # not echoed by current API
                "error_msg": "",
                "raw": resp,
            }
        # Everything else = rejected (book moved away, etc.).
        return _rejected_order(
            error_msg or f"no fill (status={api_status}, success={success})",
            order_id=order_id,
            raw=resp,
        )

    # ============================================================
    #  Settlement (Gamma)
    # ============================================================

    def read_settlement(self, slug: str) -> dict | None:
        """After a 15min window closes, check whether Polymarket has resolved it.

        Returns:
            {"resolved": True, "winning_outcome": "UP"|"DOWN", "resolved_at_iso": str}
              -> the market is settled and one side has paid out.
            None
              -> not resolved yet, or transient API error (caller should retry next cycle).

        Note: Polymarket's UMA oracle typically resolves within seconds of window
        close, but can take up to a few minutes. The caller has a fallback (Bitstamp
        candle) for "stuck" cases - see live.py's pending-bet handler.
        """
        url = f"{self.gamma_host}/markets/slug/{slug}"
        try:
            r = requests.get(url, timeout=self.timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            market = r.json()
            if not market or not market.get("closed"):
                return None

            outcomes = market.get("outcomes", '["Up","Down"]')
            outcome_prices = market.get("outcomePrices", '["0","0"]')
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices)

            # Winner = the outcome whose settled price == "1".
            winner_idx = next(
                (i for i, p in enumerate(outcome_prices) if str(p) in ("1", "1.0")),
                None,
            )
            if winner_idx is None:
                return None
            winning = "UP" if str(outcomes[winner_idx]).lower() == "up" else "DOWN"
            return {
                "resolved": True,
                "winning_outcome": winning,
                "resolved_at_iso": market.get("endDate"),
            }
        except (requests.exceptions.RequestException, ValueError) as e:
            logger.warning("read_settlement failed for slug=%s: %s", slug, e)
            return None

    # ============================================================
    #  Redemption (on-chain)
    # ============================================================

    def redeem(self, condition_id: str) -> str | None:
        """Call CTF.redeemPositions(pUSD, 0x00...00, condition_id, [1, 2]) on-chain.

        Both index sets ([1, 2]) are submitted because we don't track which side
        we held - the CTF contract only credits the winning one anyway.

        Returns:
            tx_hash (str) on confirmed submission, OR
            None on any failure (insufficient gas, revert, signing error, ...).

        Never raises - failures here mean shares stay on-chain until next attempt.
        The bet stays accounted-for in bets.csv with redeem_tx_hash=null.
        """
        try:
            parent_collection = bytes(32)   # all-zeros (Polymarket convention)
            cond_bytes = bytes.fromhex(condition_id[2:] if condition_id.startswith("0x") else condition_id)
            if len(cond_bytes) != 32:
                logger.error("redeem: condition_id has wrong length: %s", condition_id)
                return None

            nonce = self.web3.eth.get_transaction_count(self.eoa_address)
            tx = self.ctf.functions.redeemPositions(
                PUSD_POLYGON_ADDRESS,
                parent_collection,
                cond_bytes,
                [1, 2],
            ).build_transaction({
                "from": self.eoa_address,
                "nonce": nonce,
                "gas": 300_000,
                "maxFeePerGas": self.web3.eth.gas_price * 2,
                "maxPriorityFeePerGas": self.web3.to_wei(30, "gwei"),
            })
            signed = self.web3.eth.account.sign_transaction(tx, self._private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed.raw_transaction)
            tx_hash_hex = tx_hash.hex()
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            if receipt.status != 1:
                logger.error("redeem reverted on-chain: tx=%s", tx_hash_hex)
                return None
            logger.info("redeem ok: tx=%s condition=%s", tx_hash_hex, condition_id)
            return tx_hash_hex
        except Exception as e:
            logger.exception("redeem failed for condition=%s: %s", condition_id, e)
            return None


if __name__ == "__main__":
    # Smoke test: read balances + look up the next 15-min market.
    # Requires .env with LIVE_MODE-related vars populated.
    import sys
    from datetime import datetime, timezone

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        client = PolymarketClient.from_env()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"wallet_eoa = {client.eoa_address}")
    print(f"proxy      = {client.proxy_address}")
    print(f"usdc_balance  = {client.get_usdc_balance():.4f}")
    print(f"matic_balance = {client.get_matic_balance():.6f}")

    # Next 15-min boundary.
    now_s = int(time.time())
    next_boundary = ((now_s // 900) + 1) * 900
    print(f"next 15m boundary: {next_boundary} = "
          f"{datetime.fromtimestamp(next_boundary, tz=timezone.utc).isoformat()}")
    try:
        market = client.find_btc_15m_market(next_boundary)
        print(f"market: {market}")
    except MarketNotFoundError as e:
        print(f"market not found: {e}")
