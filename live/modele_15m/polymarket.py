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
key via py-clob-client-v2 at boot, kept in memory only.

SDK: py-clob-client-v2. Polymarket migrated to CLOB V2 on 2026-04-28; the old
py-clob-client v1 returns `order_version_mismatch` on every order against
production and is dead. V2 keeps signature_type=2 (POLY_GNOSIS_SAFE) for the
proxy-wallet flow.

See docs/superpowers/specs/2026-05-14-polymarket-live-betting-design.md
for the full design + the empirical findings that drove these choices.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests
from eth_account import Account
from py_clob_client_v2 import (
    ClobClient,
    MarketOrderArgsV2,
    OrderType,
    Side,
)
from web3 import Web3

logger = logging.getLogger("live.polymarket")

# --- Polygon constants ---

# USDC.e (the native USDC on Polygon, 6 decimals) - what the proxy wallet holds.
USDC_POLYGON_ADDRESS = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
USDC_DECIMALS = 6

# pUSD - Polymarket's 1:1 wrapper of USDC.e used as CTF collateral. redeemPositions
# returns pUSD; the proxy Safe should auto-unwrap to USDC.e (to verify at smoke test).
PUSD_POLYGON_ADDRESS = Web3.to_checksum_address("0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB")

# CLOB V2 redemption contracts (used by redeem() via the relayer).
CTF_CONTRACT_ADDRESS    = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
CTF_COLLATERAL_ADAPTER  = Web3.to_checksum_address("0xAdA100Db00Ca00073811820692005400218FcE1f")
# The deposit-wallet relayer forwards batches via the DepositWalletFactory.
DEPOSIT_WALLET_FACTORY  = Web3.to_checksum_address("0x00000000000Fb5C9ADea0298D729A0CB3823Cc07")

# Polymarket relayer REST API - submits gasless deposit-wallet batches on-chain.
DEFAULT_RELAYER_HOST = "https://relayer-v2.polymarket.com"

CHAIN_ID_POLYGON = 137
# SignatureTypeV2.POLY_1271 == 3 = the CLOB V2 "deposit wallet" flow. The EOA
# signs; the funder is the deposit wallet that holds the pUSD. Verified
# empirically 2026-05-20: signature_type=2 (Gnosis Safe) is rejected by CLOB V2
# with "maker address not allowed, please use the deposit wallet flow", while
# signature_type=3 + funder=<proxy> resolves to the 50 pUSD balance.
SIGNATURE_TYPE_DEPOSIT = 3

# Min order size imposed by Polymarket (in shares). Below this the CLOB rejects.
POLYMARKET_MIN_SHARES = 5

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

# ConditionalTokens (ERC-1155) - isApprovedForAll / setApprovalForAll, used to
# check / grant the redeem adapter operator rights over the deposit wallet.
CTF_ABI = [
    {"name": "isApprovedForAll", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "owner", "type": "address"},
                {"name": "operator", "type": "address"}],
     "outputs": [{"name": "", "type": "bool"}]},
    {"name": "setApprovalForAll", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "operator", "type": "address"},
                {"name": "approved", "type": "bool"}],
     "outputs": []},
]

# CtfCollateralAdapter - redeemPositions burns the resolved outcome tokens and
# returns pUSD. parentCollectionId is always 0x00..00 for Polymarket.
ADAPTER_ABI = [
    {"name": "redeemPositions", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "collateralToken", "type": "address"},
                {"name": "parentCollectionId", "type": "bytes32"},
                {"name": "conditionId", "type": "bytes32"},
                {"name": "indexSets", "type": "uint256[]"}],
     "outputs": []},
]

# EIP-712 types for signing a DepositWallet batch. The relayer reconstructs the
# Batch from the /submit payload and verifies this signature against it.
_DEPOSIT_WALLET_EIP712_TYPES = {
    "Call": [
        {"name": "target", "type": "address"},
        {"name": "value", "type": "uint256"},
        {"name": "data", "type": "bytes"},
    ],
    "Batch": [
        {"name": "wallet", "type": "address"},
        {"name": "nonce", "type": "uint256"},
        {"name": "deadline", "type": "uint256"},
        {"name": "calls", "type": "Call[]"},
    ],
}


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
        relayer_host: str = DEFAULT_RELAYER_HOST,
        relayer_api_key: str | None = None,
        relayer_api_key_address: str | None = None,
    ):
        self.eoa_address = Web3.to_checksum_address(eoa_address)
        self.proxy_address = Web3.to_checksum_address(proxy_address)
        self.gamma_host = gamma_host.rstrip("/")
        self.timeout = timeout
        self.slippage_bps = slippage_bps
        self._private_key = private_key   # also used to sign redeem batches

        # Relayer creds for auto-redeem. Optional: if absent, redeem() falls back
        # to the manual sentinel. The key is created at polymarket.com ->
        # Settings -> API Keys (self-service, no Builder Program needed).
        self.relayer_host = relayer_host.rstrip("/")
        self.relayer_api_key = relayer_api_key
        self.relayer_api_key_address = (
            Web3.to_checksum_address(relayer_api_key_address)
            if relayer_api_key_address else None
        )

        # web3 client for direct Polygon reads (balances, redeem pre-checks).
        self.web3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": timeout}))
        # USDC.e and pUSD are BOTH relevant. Users deposit USDC.e and Polymarket
        # auto-wraps it into pUSD (its 1:1 internal collateral) on the proxy.
        # The CLOB charges orders against pUSD. Smoke-tested 2026-05-14: a funded
        # account shows 0 USDC.e + 50 pUSD on the proxy. We read BOTH and sum
        # so we capture either case (mid-wrap, manual unwrap, etc.).
        self.usdc_e = self.web3.eth.contract(address=USDC_POLYGON_ADDRESS, abi=ERC20_ABI)
        self.pusd = self.web3.eth.contract(address=PUSD_POLYGON_ADDRESS, abi=ERC20_ABI)
        # Contracts for the redeem flow: the CTF ERC-1155 + the redeem adapter.
        self.ctf = self.web3.eth.contract(address=CTF_CONTRACT_ADDRESS, abi=CTF_ABI)
        self.adapter = self.web3.eth.contract(address=CTF_COLLATERAL_ADAPTER, abi=ADAPTER_ABI)

        # CLOB V2 client. signature_type=3 = the "deposit wallet" flow: the EOA
        # signs, the funder is the deposit wallet that holds the pUSD.
        self.clob = ClobClient(
            host=clob_host,
            chain_id=CHAIN_ID_POLYGON,
            key=private_key,
            signature_type=SIGNATURE_TYPE_DEPOSIT,
            funder=self.proxy_address,
        )
        # L2 API credentials are derived from the private key. Kept in memory only.
        # create_or_derive_api_key() always tries POST /auth/api-key first; for an
        # account whose key already exists that returns a 400 the SDK logs at
        # ERROR before falling back to GET /auth/derive-api-key (which succeeds).
        # That benign 400 would otherwise spam the error bot at every boot, so we
        # silence the SDK's http-helper logger just for the duration of this call.
        _sdk_log = logging.getLogger("py_clob_client_v2.http_helpers.helpers")
        _prev_level = _sdk_log.level
        _sdk_log.setLevel(logging.CRITICAL)
        try:
            self.clob.set_api_creds(self.clob.create_or_derive_api_key())
        finally:
            _sdk_log.setLevel(_prev_level)

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
            relayer_host=os.environ.get("RELAYER_HOST", DEFAULT_RELAYER_HOST),
            # Optional - enables auto-redeem. Create the key at polymarket.com
            # -> Settings -> API Keys. Without it, redeem() stays manual.
            relayer_api_key=os.environ.get("RELAYER_API_KEY") or None,
            relayer_api_key_address=(os.environ.get("RELAYER_API_KEY_ADDRESS")
                                     or os.environ.get("POLYGON_EOA_ADDRESS")),
        )

    # ============================================================
    #  Balances (direct Polygon RPC reads, no Polymarket API call)
    # ============================================================

    def get_usdc_balance(self) -> float:
        """Free trading balance on the Polymarket proxy wallet, in human units.

        Sums USDC.e + pUSD. Polymarket auto-wraps deposited USDC.e to pUSD
        (its 1:1 internal collateral) and the CLOB charges orders against pUSD.
        Empirically a freshly-funded account shows 0 USDC.e and N pUSD on the
        proxy; reading both covers the rare case of unwrapped USDC.e too.
        """
        wei_usdc_e = self.usdc_e.functions.balanceOf(self.proxy_address).call()
        wei_pusd = self.pusd.functions.balanceOf(self.proxy_address).call()
        return (wei_usdc_e + wei_pusd) / (10 ** USDC_DECIMALS)

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

    def place_fak_order(self, token_id: str, usdc_amount: float,
                        dry_run: bool = False) -> dict:
        """Place a Fill-And-Kill market BUY on `token_id` for `usdc_amount` USDC.

        Uses the CLOB V2 market-order primitive: amount is USDC to spend, the
        engine matches against the book at the best available prices and kills
        whatever can't fill (FAK). `amount` bounds the dollar spend.

        `dry_run=True` builds + signs the order (real EIP-712 signature via
        clob.create_market_order) but does NOT post it.

        Polymarket enforces a 5-share minimum; an `amount` too small to buy 5
        shares is rejected by the CLOB and surfaces here as status="rejected".

        Returns:
            {order_id, status, fill_price_avg, shares_filled, usdc_spent,
             fee_usdc, error_msg, raw}
            status in {"filled", "rejected", "failed", "dry_run"}.
        """
        args = MarketOrderArgsV2(
            token_id=token_id,
            amount=float(usdc_amount),
            side=Side.BUY,
            order_type=OrderType.FAK,
        )

        # Dry-run: build + sign only. create_market_order does read-only GETs
        # (tick-size / neg-risk) + a local EIP-712 signature - side-effect-free,
        # so a retry on a transient error is safe.
        if dry_run:
            for attempt in range(2):
                try:
                    self.clob.create_market_order(args)   # real signing, no POST
                    logger.info("DRY-RUN: market FAK order signed but NOT posted "
                                "(token=%s amount=%.2f USDC)", token_id, usdc_amount)
                    return {
                        "order_id": None,
                        "status": "dry_run",
                        "fill_price_avg": None,
                        "shares_filled": 0.0,
                        "usdc_spent": 0.0,
                        "fee_usdc": None,
                        "error_msg": "",
                        "raw": {"dry_run": True, "amount_usdc": usdc_amount,
                                "signed_ok": True},
                    }
                except Exception as e:
                    if attempt == 0:
                        logger.warning("create_market_order failed (attempt 1/2): %s", e)
                        time.sleep(1)
                        continue
                    logger.error("create_market_order failed twice: %s", e)
                    return _failed_order(f"create_market_order: {e}")

        # Real order: build + sign + POST in one call. SINGLE attempt - no
        # blind retry, since a retry would mint a fresh order and risk a
        # double-place. FAK never rests on the book.
        try:
            resp = self.clob.create_and_post_market_order(
                args, order_type=OrderType.FAK,
            )
            return self._interpret_order_response(resp)
        except Exception as e:
            logger.exception("create_and_post_market_order failed: %s", e)
            return _failed_order(f"create_and_post_market_order: {e}")

    @staticmethod
    def _interpret_order_response(resp: Any) -> dict:
        """Normalize the CLOB V2 market-order response into our standard shape.

        The V2 response shape is handled defensively: it may be a dict or an
        SDK object, and key names vary. The raw response is always kept in the
        returned dict under "raw" so unexpected shapes are debuggable from the
        bets.csv / logs.
        """
        # Coerce SDK objects to a dict view.
        if isinstance(resp, dict):
            d = resp
        else:
            d = getattr(resp, "__dict__", None) or {}
            # Some SDK objects expose a .dict()/.to_dict()
            for meth in ("dict", "to_dict", "model_dump"):
                fn = getattr(resp, meth, None)
                if callable(fn):
                    try:
                        d = fn()
                        break
                    except Exception:
                        pass

        def _first(*keys):
            for k in keys:
                if k in d and d[k] not in (None, ""):
                    return d[k]
            return None

        order_id = _first("orderID", "order_id", "id", "orderId")
        error_msg = _first("errorMsg", "error", "error_message") or ""
        api_status = _first("status", "state")
        success = bool(d.get("success", api_status in ("matched", "filled", "live", "success")))

        shares_filled = _parse_amount(_first("takingAmount", "sizeMatched",
                                             "shares_filled", "size_matched", "filledSize"))
        usdc_spent = _parse_amount(_first("makingAmount", "usdc_spent",
                                          "amountSpent", "filledAmount"))
        fee = _parse_amount(_first("fee", "feeUsdc", "fee_usdc")) or None

        if shares_filled > 0:
            avg_price = usdc_spent / shares_filled if shares_filled > 0 else None
            return {
                "order_id": order_id,
                "status": "filled",
                "fill_price_avg": avg_price,
                "shares_filled": shares_filled,
                "usdc_spent": usdc_spent,
                "fee_usdc": fee,
                "error_msg": "",
                "raw": d,
            }
        # No shares matched -> rejected (book too thin, below 5-share min, etc.).
        return _rejected_order(
            error_msg or f"no fill (status={api_status}, success={success})",
            order_id=order_id,
            raw=d,
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

    # Sentinel returned by redeem() when auto-redeem is not configured (no
    # RELAYER_API_KEY). Stored in bets.csv:redeem_tx_hash so the row is auditable
    # and the operator knows the win must be claimed by hand.
    REDEEM_MANUAL = "MANUAL_REDEEM_REQUIRED"

    def _relayer_headers(self) -> dict:
        return {
            "RELAYER_API_KEY": self.relayer_api_key,
            "RELAYER_API_KEY_ADDRESS": self.relayer_api_key_address,
            "Content-Type": "application/json",
        }

    def redeem(self, condition_id: str) -> str | None:
        """Redeem a resolved position for `condition_id` into pUSD, via the
        Polymarket relayer (gasless, deposit-wallet flow).

        HOW IT WORKS
        ------------
        Winning CTF ERC-1155 shares sit in the deposit wallet (a smart contract).
        Redeeming runs the call AS the deposit wallet, which is operator-gated:
        only Polymarket's relayer may submit it. So we:
          1. GET {relayer}/nonce?address=<eoa>&type=WALLET   - the batch nonce.
          2. Build a Batch of calls: adapter.redeemPositions(pUSD, 0x0,
             conditionId, [1,2]); prepend CTF.setApprovalForAll(adapter, true)
             if the adapter is not yet an approved operator.
          3. EIP-712-sign the Batch with the owner key (domain "DepositWallet").
          4. POST {relayer}/submit with type=WALLET + depositWalletParams.
          5. Poll GET {relayer}/transaction until STATE_CONFIRMED.

        Returns the confirmed on-chain tx hash on success. Returns
        REDEEM_MANUAL if no relayer key is configured. Returns None on any
        failure (the win stays recorded; the operator can claim by hand).
        Never raises.
        """
        if not self.relayer_api_key:
            logger.warning(
                "redeem(%s): RELAYER_API_KEY not set - auto-redeem disabled, "
                "claim this win manually at polymarket.com.", condition_id)
            return self.REDEEM_MANUAL
        try:
            return self._redeem_via_relayer(condition_id)
        except Exception as e:
            logger.exception("redeem(%s) failed: %s", condition_id, e)
            return None

    def _redeem_via_relayer(self, condition_id: str) -> str | None:
        # --- conditionId -> 32 bytes ---
        h = condition_id[2:] if condition_id.startswith("0x") else condition_id
        cond = bytes.fromhex(h)
        if len(cond) != 32:
            logger.error("redeem: conditionId not 32 bytes: %s", condition_id)
            return None

        # --- 1. relayer nonce ---
        r = requests.get(f"{self.relayer_host}/nonce",
                         params={"address": self.eoa_address, "type": "WALLET"},
                         headers=self._relayer_headers(), timeout=self.timeout)
        r.raise_for_status()
        nonce = int(r.json()["nonce"])

        # --- 2. build the batch of calls ---
        calls: list[dict] = []
        approved = self.ctf.functions.isApprovedForAll(
            self.proxy_address, CTF_COLLATERAL_ADAPTER).call()
        if not approved:
            approve_data = self.ctf.encode_abi(
                "setApprovalForAll", args=[CTF_COLLATERAL_ADAPTER, True])
            calls.append({"target": CTF_CONTRACT_ADDRESS, "value": "0",
                          "data": approve_data})
            logger.info("redeem: adapter not approved - prepending setApprovalForAll")
        redeem_data = self.adapter.encode_abi(
            "redeemPositions",
            args=[PUSD_POLYGON_ADDRESS, b"\x00" * 32, cond, [1, 2]])
        calls.append({"target": CTF_COLLATERAL_ADAPTER, "value": "0",
                      "data": redeem_data})

        # --- 3. EIP-712-sign the Batch ---
        deadline = int(time.time()) + 600
        typed = {
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
                **_DEPOSIT_WALLET_EIP712_TYPES,
            },
            "primaryType": "Batch",
            "domain": {"name": "DepositWallet", "version": "1",
                       "chainId": CHAIN_ID_POLYGON,
                       "verifyingContract": self.proxy_address},
            "message": {
                "wallet": self.proxy_address,
                "nonce": nonce,
                "deadline": deadline,
                "calls": [
                    {"target": Web3.to_checksum_address(c["target"]),
                     "value": int(c["value"]),
                     "data": (c["data"] if isinstance(c["data"], (bytes, bytearray))
                              else bytes.fromhex(c["data"][2:] if c["data"].startswith("0x")
                                                 else c["data"]))}
                    for c in calls
                ],
            },
        }
        signature = Account.sign_typed_data(
            self._private_key, full_message=typed).signature.hex()
        if not signature.startswith("0x"):
            signature = "0x" + signature

        # --- 4. submit to the relayer ---
        payload = {
            "type": "WALLET",
            "from": self.eoa_address,
            "to": DEPOSIT_WALLET_FACTORY,
            "nonce": str(nonce),
            "signature": signature,
            "depositWalletParams": {
                "depositWallet": self.proxy_address,
                "deadline": str(deadline),
                "calls": [{"target": Web3.to_checksum_address(c["target"]),
                           "value": str(c["value"]),
                           "data": (c["data"] if c["data"].startswith("0x")
                                    else "0x" + c["data"])}
                          for c in calls],
            },
        }
        sr = requests.post(f"{self.relayer_host}/submit", json=payload,
                           headers=self._relayer_headers(), timeout=self.timeout)
        if sr.status_code >= 400:
            logger.error("redeem: relayer /submit %s: %s", sr.status_code, sr.text[:300])
            return None
        tx_id = sr.json().get("transactionID")
        if not tx_id:
            logger.error("redeem: relayer /submit gave no transactionID: %s", sr.json())
            return None
        logger.info("redeem: relayer accepted batch, transactionID=%s", tx_id)

        # --- 5. poll until confirmed ---
        # GET /transaction returns a LIST of transaction entries; pick the one
        # whose transactionID matches and read its `state` + `transactionHash`.
        for _ in range(40):                       # ~40 * 3s = 2 min
            time.sleep(3)
            tr = requests.get(f"{self.relayer_host}/transaction",
                              params={"id": tx_id}, headers=self._relayer_headers(),
                              timeout=self.timeout)
            if tr.status_code >= 400:
                continue
            body = tr.json()
            entries = body if isinstance(body, list) else [body]
            entry = next((e for e in entries
                          if str(e.get("transactionID")) == str(tx_id)), None)
            if entry is None:
                continue
            state = entry.get("state", "")
            if state == "STATE_CONFIRMED":
                tx_hash = entry.get("transactionHash") or tx_id
                logger.info("redeem ok: condition=%s tx=%s", condition_id, tx_hash)
                return tx_hash
            if state in ("STATE_FAILED", "STATE_REVERTED", "STATE_CANCELLED"):
                logger.error("redeem: relayer tx %s ended %s", tx_id, state)
                return None
        logger.error("redeem: relayer tx %s not confirmed after polling", tx_id)
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
