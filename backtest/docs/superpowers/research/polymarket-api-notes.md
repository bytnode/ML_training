# Polymarket Public API — Reference Notes for a 15-min BTC Up/Down Bot

Captured 2026-05-14 from live API + official docs. Sources cited inline.
Target: place a real BUY on the YES side (Up or Down) of the matching 15-min Polymarket BTC market each time our model emits a decision at xx:00/15/30/45 UTC.

---

## 0. Headline findings — read before designing the wiring

1. **The 15-minute BTC market exists, is one binary market per 15-min window, and is keyed deterministically by the *start-of-window* UNIX timestamp.** Slug: `btc-updown-15m-<unix_ts>` where `unix_ts` is the start of the window in seconds (UTC, aligned to xx:00/15/30/45 → multiple of 900). Confirmed live below.
   - https://polymarket.com/event/btc-updown-15m-1778697900
   - https://gamma-api.polymarket.com/markets/slug/btc-updown-15m-1778779800

2. **Resolution source is Chainlink BTC/USD data stream, *not* Bitstamp / Binance.** The model we trained on Bitstamp 15m candles will not be perfectly aligned with the market's resolution price. Treat this as the single biggest risk for the live PnL.
   - https://data.chain.link/streams/btc-usd
   - Quote from a live market description: *"This market will resolve to 'Up' if the Bitcoin price at the end of the time range specified in the title is greater than or equal to the price at the beginning of that range. Otherwise, it will resolve to 'Down'. The resolution source for this market is information from Chainlink, specifically the BTC/USD data stream available at https://data.chain.link/streams/btc-usd."*

3. **`>=` resolves to Up** — a flat tick is Up. Worth tightening the model's training label to match.

4. **Polymarket geo-blocks France and most of Western Europe** for new orders. A server in eu-west-2 / Dublin / Frankfurt with a residential or non-FR EU IP may still get through, but ToS §2.1.4 prohibits VPNs. A docker host on a US/FR IP will be rejected by `/api/geoblock`. Treat this as a deployment-location decision.
   - https://docs.polymarket.com/api-reference/geoblock
   - https://help.polymarket.com/en/articles/13364163-geographic-restrictions

5. **`py-clob-client` (v1) is archived as of late 2025**. New code should target **`py-clob-client-v2`** (latest 1.0.1, 2026-05-09). v1 still works but does not support the new `POLY_1271` deposit-wallet flow used by Polymarket since the Pectra/EIP-7702 rollout.
   - https://github.com/Polymarket/py-clob-client (banner: "archived")
   - https://github.com/Polymarket/py-clob-client-v2
   - https://pypi.org/project/py-clob-client-v2/

6. **There is no public testnet for these specific 15-min markets.** The CLOB code can be pointed at Polygon Amoy chain_id=80002 with a different exchange address, but the 15-min crypto series is mainnet-only. End-to-end dry-runs need to be done against mainnet with tiny size (the `minimum_order_size` for these markets is **5 shares**, i.e. about $2.50 at mid-50¢ — see §3.4).

7. **Fee model on the 15-min crypto markets right now: `feeSchedule = {exponent:1, rate:0.07, takerOnly:true, rebateRate:0.2}`**. That is `fee = 0.07 * size * p * (1-p)` charged on takers only, capped to 7% at p=0.5 and decreasing symmetrically toward 0/1. Confirmed live in the JSON below.

8. **The host clock matters.** L2 HMAC signature includes `POLY_TIMESTAMP` in seconds. Drift > a few seconds → `401`. Hit `GET https://clob.polymarket.com/time` (returns a plain integer of seconds) at boot and align.

---

## 1. Market discovery — finding "the current cycle" market

### 1.1 Slug & event structure

Confirmed live response (raw, abbreviated, formatting preserved):

```bash
$ curl -s "https://gamma-api.polymarket.com/markets/slug/btc-updown-15m-1778779800"
```
```json
{
  "id": "2252287",
  "question": "Bitcoin Up or Down - May 14, 1:30PM-1:45PM ET",
  "conditionId": "0xf63cd3b84f44620798f24fc7e79446af21beee4d6b1be2122325abf0ce377bf2",
  "slug": "btc-updown-15m-1778779800",
  "resolutionSource": "https://data.chain.link/streams/btc-usd",
  "endDate": "2026-05-14T17:45:00Z",
  "startDate": "2026-05-13T17:37:48.501906Z",
  "outcomes": "[\"Up\", \"Down\"]",
  "outcomePrices": "[\"0.495\", \"0.505\"]",
  "questionID": "0xf59477203192248f3243ac54c1bb18125979f3f00541e3c5261db0e732b50b6b",
  "enableOrderBook": true,
  "orderPriceMinTickSize": 0.01,
  "orderMinSize": 5,
  "clobTokenIds": "[\"100509372309051301771210562424047876079302122434803645363811103128351182475722\", \"97096356311969210456624340956011480897647519953018486625533382999976302647828\"]",
  "makerBaseFee": 1000,
  "takerBaseFee": 1000,
  "acceptingOrders": true,
  "negRisk": false,
  "events": [{ "id": "481744", "ticker": "btc-updown-15m-1778779800",
               "slug": "btc-updown-15m-1778779800",
               "seriesSlug": "btc-up-or-down-15m",
               "startTime": "2026-05-13T17:42:20.519272Z",
               "endDate": "2026-05-14T17:45:00Z",
               "negRisk": false,
               "automaticallyResolved": true,
               "automaticallyActive": true,
               "eventMetadata": { "finalPrice": null, "priceToBeat": null } }],
  "spread": 0.01, "bestBid": 0.49, "bestAsk": 0.50,
  "feesEnabled": true,
  "feeType": "crypto_fees_v2",
  "feeSchedule": {"exponent":1, "rate":0.07, "takerOnly":true, "rebateRate":0.2}
}
```

Important takeaways from that JSON:

| Field | Meaning | Notes |
|---|---|---|
| `slug` | `btc-updown-15m-<unix_ts_seconds>` | `1778779800 = 2026-05-14 17:30:00 UTC` — start of the 15-min window. |
| `conditionId` | bytes32 hash of the market | Needed for redemption (`CTFExchange.redeemPositions`). |
| `questionID` | bytes32 question hash | Used by the oracle, not by the trader. |
| `clobTokenIds` | **JSON-encoded string** of `[UP, DOWN]` | Must `json.loads()` it. Order is positional: index 0 = "Up", index 1 = "Down" (verify with the `outcomes` field every time — see warning below). |
| `outcomes` | `'["Up", "Down"]'` (also JSON-encoded string) | Always parse and verify; do **not** assume index→outcome mapping. |
| `outcomePrices` | `'["0.495","0.505"]'` | Latest auto-quoted price per outcome. Use the order book for real bid/ask. |
| `enableOrderBook` | true | If false the market is AMM-only and cannot be traded via CLOB. |
| `orderPriceMinTickSize` | 0.01 | i.e. `tickSize="0.01"` — needed on every order. |
| `orderMinSize` | 5 | Minimum **shares** per order, not dollars. At p≈0.5 → ~$2.50. |
| `negRisk` | false | Binary up/down markets are non-neg-risk; the regular CTF Exchange handles them. |
| `acceptingOrders` | true | False = window has elapsed / no new orders accepted. |
| `feeSchedule.rate` | 0.07 | 7% taker-only fee at p=0.5 (formula §3.6). |

Sample of the active event-level structure also confirms `seriesSlug: "btc-up-or-down-15m"` — every 15-min Bitcoin market is a child event of that series.

**Series of related slugs** (verified via `polymarket.com/crypto/15M`):
- `btc-updown-15m-<ts>` — Bitcoin
- `eth-updown-15m-<ts>` — Ethereum
- `sol-updown-15m-<ts>` — Solana
- `xrp-updown-15m-<ts>` — XRP
- `doge-updown-15m-<ts>` — Dogecoin
- `bnb-updown-15m-<ts>` — BNB
- `hype-updown-15m-<ts>` — Hyperliquid

(5-min variants exist too: `btc-updown-5m-<ts>`.)

URL on the frontend is `https://polymarket.com/event/btc-updown-15m-<ts>`.

References:
- https://polymarket.com/crypto/15M
- https://github.com/handiko/Polymarket-Market-Finder — third-party "find me the current 15m market" utility, same slug formula
- https://github.com/Polymarket/py-clob-client/issues/244 — bug report from someone who got the timestamp formula wrong; confirms the format is start-of-window in seconds

### 1.2 How to query the market for "the next 15-min UTC boundary"

Two equivalent endpoints (no auth needed):

```bash
# (a) Path form — clean 404 if not found
curl https://gamma-api.polymarket.com/markets/slug/btc-updown-15m-<ts>

# (b) Query form — returns [] if not found (note: trailing array)
curl "https://gamma-api.polymarket.com/markets?slug=btc-updown-15m-<ts>"
```
Endpoint reference: https://docs.polymarket.com/api-reference/markets/get-market-by-slug

Helper to compute the slug for the upcoming window:

```python
import time

def next_15m_slug(asset: str = "btc") -> tuple[str, int]:
    """Return (slug, window_start_unix) for the *upcoming* 15-min window.
    Decision is made at the close of the previous candle, so we bet on the next one."""
    now = int(time.time())
    # Floor to 15-min boundary, then add 15 min for the next window
    next_ts = (now // 900) * 900 + 900
    return f"{asset}-updown-15m-{next_ts}", next_ts
```

A defensive variant probes ±1 window in case of clock drift / market not yet seeded:

```python
def find_active_market(asset: str = "btc") -> dict:
    import requests
    base = "https://gamma-api.polymarket.com/markets/slug"
    now = int(time.time())
    candidates = [(now // 900) * 900 + 900, (now // 900) * 900, (now // 900) * 900 + 1800]
    for ts in candidates:
        r = requests.get(f"{base}/{asset}-updown-15m-{ts}", timeout=2)
        if r.status_code == 200 and r.json():
            m = r.json()
            if m.get("active") and m.get("acceptingOrders") and m.get("enableOrderBook"):
                return m
    raise RuntimeError("no active 15m market found")
```

### 1.3 Identifiers to keep in our log

Per the live JSON, persist at least:

| Local field | Source field | Type | Use |
|---|---|---|---|
| `gamma_market_id` | `id` | string ("2252287") | Polymarket internal — handy for cross-ref |
| `condition_id` | `conditionId` | 0x… bytes32 | Required for **redemption** post-resolution and is the canonical onchain id |
| `slug` | `slug` | string | Human-readable + cache key |
| `event_id` | `events[0].id` | string | Series-level grouping |
| `token_id_up` | `json.loads(clobTokenIds)[idx_of("Up")]` | uint256 decimal string | Used in order `tokenId` for BUY-Up |
| `token_id_down` | `json.loads(clobTokenIds)[idx_of("Down")]` | uint256 decimal string | Used in order `tokenId` for BUY-Down |
| `tick_size` | `orderPriceMinTickSize` | float (0.01) | Required option when signing |
| `min_size` | `orderMinSize` | int (5) | Min shares; reject before signing |
| `neg_risk` | `negRisk` | bool (false) | Required option; routes to NegRisk exchange if true |
| `end_date_iso` | `endDate` | RFC3339 | Window close time |
| `accepting_orders` | `acceptingOrders` | bool | If false, skip — Polymarket freezes the book ~1-2s before window end |

**Position-mapping safety net** — always derive UP/DOWN from `outcomes`, never from index assumption:

```python
import json
outcomes  = json.loads(market["outcomes"])        # ["Up", "Down"]  (in practice)
token_ids = json.loads(market["clobTokenIds"])    # ["123…", "456…"]
token_up   = token_ids[outcomes.index("Up")]
token_down = token_ids[outcomes.index("Down")]
```

### 1.4 Rate limits for discovery (Gamma)

From https://docs.polymarket.com/api-reference/rate-limits :

| Endpoint family | Limit (per 10 s) |
|---|---|
| Gamma general | 4,000 |
| `/events` | 500 |
| `/markets` | 300 |
| listing combined | 900 |
| `/comments` | 200 |
| `/tags` | 200 |
| `/public-search` | 350 |

For one bet per 15 minutes we're nowhere near these. Even a 10-Hz orderbook poll inside a window is fine (1,500/10s on `/book`).

---

## 2. CLOB authentication — L1 (wallet) and L2 (API key)

Authoritative source: https://docs.polymarket.com/developers/CLOB/authentication
Plus PolymarketAuthTroubleshooting: https://agentbets.ai/guides/polymarket-auth-troubleshooting/

### 2.1 Big picture

Polymarket's CLOB uses a two-tier model:

- **L1 (wallet)** — EIP-712 signature of a fixed "ClobAuth" message with the trader's private key. Used **once** to create or derive L2 credentials, and used **on every order** to sign the order primitive itself (the order's `signature` field).
- **L2 (HMAC)** — `apiKey` + `apiSecret` + `passphrase` derived from L1. These produce per-request `POLY_*` HMAC headers required for all stateful endpoints (post order, cancel, get orders, etc.).

Quote (https://docs.polymarket.com/developers/CLOB/authentication):
> *"Even with L2 authentication headers, methods that create user orders still require the user to sign the order payload."*

So the wallet is needed for **every** order, not just at boot.

### 2.2 L1 — what is signed

EIP-712 typed-data, captured from the SDK (https://github.com/Polymarket/py-clob-client/blob/main/py_clob_client/signing/eip712.py):

```javascript
domain = {
  name:    "ClobAuthDomain",
  version: "1",
  chainId: 137                     // 80002 on Amoy testnet
}

types = {
  ClobAuth: [
    { name: "address",   type: "address"  },
    { name: "timestamp", type: "string"   },
    { name: "nonce",     type: "uint256"  },
    { name: "message",   type: "string"   }
  ]
}

value = {
  address:   signerEoa,
  timestamp: unix_seconds_as_str,
  nonce:     0,                                              // can be incremented to rotate keys
  message:   "This message attests that I control the given wallet"
}
```

The recovered ECDSA signature is what `createOrDeriveApiKey()` POSTs to `POST /auth/api-key` to receive `{key, secret, passphrase}`.

### 2.3 L2 — HMAC scheme and headers

For every authenticated CLOB request the SDK sets these 5 headers (https://docs.polymarket.com/developers/CLOB/authentication and confirmed in agentbets troubleshooting guide):

| Header | Value |
|---|---|
| `POLY_ADDRESS` | EOA address of the **signer** (the private key), with `0x` prefix |
| `POLY_TIMESTAMP` | Current UNIX timestamp in **seconds** (string) |
| `POLY_API_KEY` | UUID returned by `create_or_derive_api_creds()` |
| `POLY_PASSPHRASE` | Passphrase returned together |
| `POLY_SIGNATURE` | base64-encoded HMAC-SHA256, see below |

There is **no `POLY_NONCE`** in the request headers — `nonce` is only used when *deriving* the API key (and is part of the EIP-712 ClobAuth payload above), not on each call.

**HMAC base string** (canonical message to sign, base64-url-safe HMAC-SHA256 of it, key = base64-decoded `secret`):

```
HMAC_SHA256( base64_decode(secret),
             POLY_TIMESTAMP + method.upper() + request_path + body_or_empty )
```

The SDK's `hmac_signer` builds exactly that and base64-encodes the digest. Verify with:
- https://github.com/Polymarket/py-clob-client/blob/main/py_clob_client/headers/headers.py

### 2.4 Proxy wallet vs EOA — `signature_type` & `funder`

Polymarket distinguishes the **signer EOA** (the private key) from the **funder** (the address that actually holds the pUSD and the CTF ERC-1155s).

`signature_type` values (https://docs.polymarket.com/developers/proxy-wallet):

| Type | Name | Funder = ? | Use case |
|---|---|---|---|
| 0 | EOA | signer EOA itself | MetaMask-style direct trading. Signer must hold pUSD + POL. |
| 1 | POLY_PROXY | Polymarket proxy contract for Magic-Link | Legacy email-login users. |
| 2 | POLY_GNOSIS_SAFE | Gnosis Safe proxy created when you connected MetaMask to polymarket.com | **Most common today** for human users — the funder is the safe address shown in profile settings, not your MetaMask address. |
| 3 | POLY_1271 | Deposit-wallet address (ERC-1271 contract signature) | **Recommended for new API users** per docs.polymarket.com — orders are validated via ERC-1271 against the deposit wallet. |

Critical pitfall:
> *"Invalid Funder Address — Passing signing key address instead of proxy wallet address … Use address from polymarket.com/settings, not MetaMask address."* — agentbets troubleshooting

Practical choice for our bot:
- If we already trade on polymarket.com via MetaMask: `signature_type=2`, `funder = <0xSafeAddressFromSettings>`, `key = <MetaMask private key>`. This is what the existing `live/.env` shape (`POLYMARKET_PROXY_ADDRESS` + `POLYGON_EOA_ADDRESS` + `POLYGON_PRIVATE_KEY`) already maps to.
- If we want pure-API-only with no website history: `signature_type=3` + a deposit wallet created via `POST /create-deposit-address` (docs index references "Create deposit addresses"). This is the path the v2 docs now favor.

Either way the env var mapping for this project is:

```
POLYGON_PRIVATE_KEY      -> ClobClient(key=)
POLYGON_EOA_ADDRESS      -> derived from key; do not pass separately
POLYMARKET_PROXY_ADDRESS -> ClobClient(funder=)   # this is the address that holds pUSD
```

References:
- https://docs.polymarket.com/developers/proxy-wallet
- https://agentbets.ai/guides/polymarket-auth-troubleshooting/

### 2.5 Bootstrapping creds (Python)

```python
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds

HOST     = "https://clob.polymarket.com"
CHAIN_ID = 137
SIG_TYPE = 2            # 2 = Gnosis Safe proxy. Use 3 for new POLY_1271 deposit wallet flow.
KEY      = os.environ["POLYGON_PRIVATE_KEY"]
FUNDER   = os.environ["POLYMARKET_PROXY_ADDRESS"]

# First boot — derive once, then cache to disk / secret store
boot = ClobClient(HOST, key=KEY, chain_id=CHAIN_ID, signature_type=SIG_TYPE, funder=FUNDER)
creds = boot.create_or_derive_api_creds()        # idempotent for same nonce=0
print(creds.api_key, creds.api_secret, creds.api_passphrase)

# Subsequent boots — pass cached creds in
client = ClobClient(
    HOST, key=KEY, chain_id=CHAIN_ID, signature_type=SIG_TYPE, funder=FUNDER,
    creds=ApiCreds(api_key=creds.api_key,
                   api_secret=creds.api_secret,
                   api_passphrase=creds.api_passphrase),
)
```
Source: https://github.com/Polymarket/py-clob-client/blob/main/README.md and agentbets reference.

### 2.6 Common L2 errors

| Error | Cause | Fix |
|---|---|---|
| `INVALID_SIGNATURE` | Wrong `signature_type`, key missing `0x`, or HMAC body mismatch | Recheck signature_type, normalize key, verify body byte-for-byte |
| `NONCE_ALREADY_USED` | Trying to re-derive creds with same nonce | Increment nonce in `create_api_key(nonce=N)` |
| `Invalid Funder Address` | Passed signer EOA as `funder` for type 1/2 | Use proxy/safe address |
| `401 Unauthorized` | Drifted clock OR creds revoked | Sync via `GET /time`; re-derive |
| 401 intermittent | Clock drift > a few seconds | Always read `client.get_server_time()` at boot |

---

## 3. Placing an order — BUY YES on a 15-min binary

### 3.1 Endpoint

REST: `POST https://clob.polymarket.com/order`
Headers: the 5 `POLY_*` headers from §2.3.
Body (JSON):
```json
{
  "order": { ... full signed Order struct, see §3.3 ... },
  "owner": "<api_key UUID>",
  "orderType": "GTC" | "GTD" | "FOK" | "FAK",
  "postOnly": false
}
```

That body shape is taken directly from `py_clob_client.utilities.order_to_json`:
```python
def order_to_json(order, owner, orderType, post_only: bool = False) -> dict:
    return {"order": order.dict(), "owner": owner,
            "orderType": orderType, "postOnly": post_only}
```
Source: https://github.com/Polymarket/py-clob-client/blob/main/py_clob_client/utilities.py

Bulk variant: `POST /orders` for up to 15 orders.

### 3.2 Order types

| Type | Behavior | Right for our use case? |
|---|---|---|
| `GTC` | Rests on the book until filled/cancelled | No — we don't want to leave a stale resting bid past the window |
| `GTD` | Same as GTC but auto-cancels at `expiration` | Workable as a safety net |
| `FOK` | Fill-Or-Kill: all-or-nothing immediate fill at the *limit price* you specified or better, else **the whole order is rejected** | Good if we want predictable cost basis; bad if liquidity at our limit is < required size |
| `FAK` | Fill-And-Kill (aka IOC): fill whatever is available at limit-or-better, **cancel the rest** | **Best fit** for "take liquidity now, accept a partial fill" |

Note `FOK` and `FAK` are still *limit* orders — the SDK calls them "market orders" but you always provide a price. To truly hit the book regardless of price, send `price=0.99` on a BUY and `FAK` (the engine will sweep up to that limit).

Source: https://docs.polymarket.com/developers/CLOB/orders/orders

**Practical recipe for our bot**:
1. Read the book (`GET /book?token_id=...`) right before posting.
2. Compute a slippage-capped limit, e.g. `limit = min(best_ask + 0.02, 0.99)`.
3. Send `FAK` at that limit. Accept partial fills; size we couldn't take stays unbought.

### 3.3 The Order struct (EIP-712)

This is the data signed by the wallet and sent inside `order` in the POST body. Full struct (https://github.com/Polymarket/ctf-exchange and confirmed via py-order-utils):

```solidity
struct Order {
    uint256 salt;             // entropy, random uint256 per order
    address maker;            // who funds — for sig_type=2/3 this is the FUNDER (proxy/safe/deposit), not the EOA
    address signer;           // EOA that signs — for sig_type=0 == maker, for 1/2/3 it's the EOA controlling the proxy
    address taker;            // 0x0000…0000 for a public order on the book
    uint256 tokenId;          // ERC-1155 id of the YES side you want to BUY (or sell)
    uint256 makerAmount;      // for BUY: USDC the maker is offering (6 decimals, i.e. 1 USDC = 1_000_000)
    uint256 takerAmount;      // for BUY: shares the maker wants in return (6 decimals)
    uint256 expiration;       // unix seconds; 0 for non-GTD (FOK/FAK/GTC) when CLOB enforces deadline server-side
    uint256 nonce;            // onchain cancel nonce, not the same as L1 ClobAuth nonce
    uint256 feeRateBps;       // accepted fee rate; must match the market's current rate or order is rejected
    uint8   side;             // 0 = BUY, 1 = SELL
    uint8   signatureType;    // 0=EOA, 1=POLY_PROXY, 2=POLY_GNOSIS_SAFE, 3=POLY_1271
    bytes   signature;        // EIP-712 of the above
}
```

Domain:
```javascript
EIP712Domain = {
  name:    "Polymarket CTF Exchange",
  version: "1",
  chainId: 137,
  verifyingContract: "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"   // CTF Exchange on Polygon mainnet
}
```
(For neg-risk markets — not our case for binary updown — the verifyingContract is the NegRisk CTF Exchange. Our markets have `negRisk: false`.)

References:
- https://github.com/Polymarket/ctf-exchange
- https://github.com/Polymarket/python-order-utils
- https://github.com/Polymarket/poly-py-eip712-structs
- Contract addresses: https://polygonscan.com/address/0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e

You should never have to construct this by hand — let py-clob-client / py-order-utils do it. But you do need to understand what's signed so the fee rate, signatureType, and funder are right.

### 3.4 Price / size encoding

This is the most error-prone area. Documented in https://nautilustrader.io/docs/latest/integrations/polymarket/ and confirmed in our live JSON:

- **Prices** are decimals in [0, 1] (i.e. probability), with the market's tick size as the increment. The binary updown 15m market uses `tick_size = 0.01` → 2 decimal places only (0.01, 0.02, …, 0.99). Tick 0.005 etc. is rejected.
- **Sizes** are decimals in **shares**, 2 decimal places. Minimum 5 shares for this market (`orderMinSize: 5`).
- **`makerAmount` / `takerAmount`** sent to the exchange are uint256 in **USDC base units (10⁶)** — these are produced by the SDK from your decimal price/size, you don't compute them yourself.
  - For a BUY of `size` shares at `price`: `makerAmount = round(price * size * 1e6)`, `takerAmount = round(size * 1e6)`.
  - For a SELL: swapped.

| Tick size | Price decimals | Size decimals | makerAmount decimals |
|---|---|---|---|
| 0.1    | 1 | 2 | 3 |
| 0.01   | 2 | 2 | 4 |
| 0.001  | 3 | 2 | 5 |
| 0.0001 | 4 | 2 | 6 |
(Source: nautilustrader docs)

The SDK rejects malformed precision with errors like `FOK order decimal places error` (https://github.com/Polymarket/py-clob-client/issues/121).

**Minimum order** for our 15m BTC market = `5 shares`. At p=0.50, that's $2.50 of YES exposure (returns $5 if won). At p=0.05, that's $0.25 ($5 payout). The general "you need $1 to play" claim from older threads is *under* the actual minimum on this specific series — budget at least $2-3.

### 3.5 Fees on the 15-min crypto market

Live `feeSchedule` returned with the market:

```json
"feeSchedule": {
  "exponent": 1,
  "rate": 0.07,
  "takerOnly": true,
  "rebateRate": 0.2
}
```

That maps to:

```
fee_in_shares = size * rate * (price * (1 - price))^exponent
               = size * 0.07 * price * (1 - price)
```
- Charged **only to the taker** (`takerOnly: true`). A resting GTC that gets matched pays nothing.
- Maker rebate = 20% of the symmetric fee, paid out via Polymarket's Maker Rebates Program.
- Peak at p=0.5 → fee = `0.07 * 0.25 = 0.0175 = 1.75%` of notional.
- At p=0.4 / 0.6 → fee = `0.07 * 0.24 = 1.68%`.
- At p=0.1 / 0.9 → fee = `0.07 * 0.09 = 0.63%`.

The field `feeRateBps` on the Order struct must match the market's current rate. The SDK reads it for you via `client.get_fee_rate_bps(token_id)` and bakes it in. **Don't hard-code 1000 bps** — the schedule above does *not* boil down to a single basis-point number; the on-chain `feeRateBps` is the peak base rate (1000 = 10%) which is then multiplied by the symmetric `p*(1-p)` curve in the contract. Confirmed live (`makerBaseFee: 1000, takerBaseFee: 1000`), and quantjourney's writeup describes the same curve.

References:
- https://docs.polymarket.com/developers/market-makers/maker-rebates-program
- https://quantjourney.substack.com/p/understanding-the-polymarket-fee
- https://nautilustrader.io/docs/latest/integrations/polymarket/

### 3.6 Partial fills

The post-order response (https://docs.polymarket.com/developers/CLOB/clients/methods-l2):

```json
{
  "success": true,
  "errorMsg": "",
  "orderID": "0xabc123…",
  "status": "matched" | "live" | "delayed" | "unmatched",
  "takingAmount":  "1234000",     // in base units, USDC for BUY
  "makingAmount":  "4900000",
  "transactionsHashes": ["0x…", …]
}
```

- `status=matched` → fully or partially filled this call. Inspect `takingAmount` (shares received) vs. requested.
- `status=unmatched` for `FOK` means rejected; for `FAK` means nothing was at our limit.
- `status=live` only for `GTC`/`GTD`.
- `transactionsHashes` is the Polygon tx hash(es) that settled the fill — useful for our PnL ledger.

The CLOB also fires user-channel WS events; see §5.

### 3.7 End-to-end Python example with py-clob-client v1

```python
import os, json, time, requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, MarketOrderArgs, OrderType, ApiCreds
from py_clob_client.order_builder.constants import BUY

HOST = "https://clob.polymarket.com"
GAMMA = "https://gamma-api.polymarket.com"

# 1. discover
ts   = (int(time.time()) // 900 + 1) * 900           # next 15-min window
slug = f"btc-updown-15m-{ts}"
m    = requests.get(f"{GAMMA}/markets/slug/{slug}", timeout=2).json()
outcomes  = json.loads(m["outcomes"])
token_ids = json.loads(m["clobTokenIds"])
TOKEN_UP   = token_ids[outcomes.index("Up")]
TOKEN_DOWN = token_ids[outcomes.index("Down")]
TICK = str(m["orderPriceMinTickSize"])               # "0.01"
NEG  = bool(m["negRisk"])                            # false for binary updown

# 2. auth (assume creds were cached earlier; first time call create_or_derive_api_creds)
client = ClobClient(
    HOST,
    key=os.environ["POLYGON_PRIVATE_KEY"],
    chain_id=137, signature_type=2,
    funder=os.environ["POLYMARKET_PROXY_ADDRESS"],
    creds=ApiCreds(
        api_key=os.environ["POLY_API_KEY"],
        api_secret=os.environ["POLY_API_SECRET"],
        api_passphrase=os.environ["POLY_PASSPHRASE"],
    ),
)

# 3. decide which side
side_token = TOKEN_UP if model_decision == "UP" else TOKEN_DOWN

# 4. read book → compute slippage-capped limit
book = client.get_order_book(side_token)
best_ask = float(book.asks[0].price) if book.asks else 0.99
limit_price = min(round(best_ask + 0.02, 2), 0.99)   # cap 2¢ slippage

# 5. send FAK (take liquidity, kill remainder)
amount_usdc = 5.00                                   # USDC to spend on this side
mo = MarketOrderArgs(token_id=side_token, amount=amount_usdc, side=BUY,
                     order_type=OrderType.FAK, price=limit_price)
signed = client.create_market_order(mo)
resp   = client.post_order(signed, OrderType.FAK)
print(resp)
```

**One quirk**: `create_market_order(amount=...)` interprets `amount` as USDC notional on BUY (good), but as **shares** on SELL — confusion-prone. We only BUY, so we're fine.

(Equivalent in v2: replace `MarketOrderArgs` import with `py_clob_client_v2.MarketOrderArgs`, add `options=PartialCreateOrderOptions(tick_size=TICK)`, and use `client.create_and_post_market_order(...)`.)

---

## 4. Order-book / mid-price discovery

### 4.1 Endpoints (CLOB, public, no auth)

| Endpoint | Sample | Notes |
|---|---|---|
| `GET /book?token_id=…` | full book, bids+asks arrays | Best for slippage-aware sizing |
| `GET /price?token_id=…&side=BUY` | best ask | `side=BUY` returns the price a buyer **would pay** — i.e. lowest ask |
| `GET /price?token_id=…&side=SELL` | best bid | the price a seller **would receive** |
| `GET /midpoint?token_id=…` | `{ "mid": "0.485" }` | float midpoint |
| `GET /spread?token_id=…` | `{ "spread": "0.01" }` | best_ask − best_bid |
| `GET /books`, `/prices`, `/midpoints` (POST body) | batch | up to many token_ids in one round-trip |

Live samples (executed 2026-05-14 17:25 UTC against `token_id=100509…475722` Up side of `btc-updown-15m-1778779800`):

```
$ curl https://clob.polymarket.com/midpoint?token_id=...   →  {"mid":"0.485"}
$ curl https://clob.polymarket.com/price?token_id=...&side=BUY   →  {"price":"0.48"}
$ curl https://clob.polymarket.com/price?token_id=...&side=SELL  →  {"price":"0.49"}
$ curl https://clob.polymarket.com/spread?token_id=...   →  {"spread":"0.01"}
```

Note the slight asymmetry: docs say "side=BUY → best ask" but live response gave 0.48 with `side=BUY` and 0.49 with `side=SELL`, suggesting the API actually returns the price *the named side gets to fill* (BUY hits best ask, but here the labels are swapped). **Always double-check with `/book` before trusting `/price`.**

Reference: documentation index at https://docs.polymarket.com/llms.txt lists "Get market price — Best bid/ask for specific token and side", "Get midpoint price", "Get spread", "Get order book".

### 4.2 Rate limits relevant to us

| Endpoint | per 10s |
|---|---|
| `/book`, `/price`, `/midpoint` | 1,500 each |
| `/books`, `/prices`, `/midpoints` | 500 each (batch) |
| `POST /order` | 5,000 burst / 48,000 per 10 min sustained |

We do one bet every 15 min → not a concern.

### 4.3 Sample full `/book` response

```json
{
  "market":"0xf63cd3b8…ce377bf2",
  "asset_id":"100509372309…475722",
  "timestamp":"1778779521134",
  "hash":"3d04cacc30699209858bc0084eabe3195a47d37c",
  "bids":[
    {"price":"0.48","size":"38.71"},
    {"price":"0.47","size":"303.76"},
    ...
  ],
  "asks":[
    {"price":"0.49","size":"153"},
    {"price":"0.50","size":"1317"},
    ...
  ]
}
```
Note: bids returned in **ascending** price order, asks in **descending** price order in this live capture. Always sort/scan defensively (`max(bids, key=price)` / `min(asks, key=price)`).

### 4.4 Slippage cap recommendation for 15m markets

These markets typically run with a 1¢ spread when liquid (verified `spread: 0.01`) and may widen to 4-6¢ in the last 30 seconds of a window. Reasonable caps for our use case:

| Time-to-close | Limit cap |
|---|---|
| > 5 min | `best_ask + 0.01` (one-tick worse) |
| 1-5 min | `best_ask + 0.02` |
| 30s-1min | `best_ask + 0.04` |
| < 30s | skip the bet (`acceptingOrders` will flip false anyway) |

The market freezes orders ~2-5 seconds before window close — check `acceptingOrders` right before posting.

---

## 5. Position / fill tracking + redemption

### 5.1 Confirming a fill

Three options, in order of operational simplicity:

1. **Inspect the `post_order` response directly** — `status=matched` plus `takingAmount` and `transactionsHashes` tell you everything for FOK/FAK. This is the simplest path for our cycle: send order → read response → log. No second request needed.

2. **Poll `GET /data/orders/{order_id}`** — returns current state (`size_matched`, `original_size`, status). Useful for `GTC` orders we leave on the book.

3. **WebSocket user channel** — `wss://ws-subscriptions-clob.polymarket.com/ws/user` streams `trade` and `order` events as they happen. Requires auth (sends an `auth` frame with `apiKey`/`secret`/`passphrase`). Overkill for one-bet-per-15-min cadence; useful if you ever batch multiple sides.

References:
- https://docs.polymarket.com/llms.txt → "Get user orders", "Get single order by ID", "WebSocket User Channel"
- https://chainstack.com/polymarket-api-for-developers/

### 5.2 Resolution and redemption

After window close:
1. The Chainlink BTC/USD stream's tick at `endDate` is fed to the oracle (`automaticallyResolved: true`, `umaResolutionStatus: "resolved"` once UMA confirms).
2. The condition gets resolved on-chain → outcome payouts written to the CTF contract.
3. **Tokens do NOT auto-burn to USDC.** Winning shares sit in the funder's ERC-1155 balance at $1 each.
4. To collect: call `CTFExchange.redeemPositions(...)` or the equivalent through the adapter, which burns the ERC-1155s for the funder and returns pUSD (a 1:1 USDC.e wrapper held at `0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB`).

Function signature (https://docs.polymarket.com/trading/ctf/redeem):

```python
redeemPositions(
    collateralToken:     address,   # pUSD: 0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB
    parentCollectionId:  bytes32,   # always 0x0000…0000 for Polymarket
    conditionId:         bytes32,   # market.conditionId
    indexSets:           uint256[], # [1, 2] = redeem both outcomes
)
```

Polygon transaction sent from the **funder** address (signer if sig_type=0; safe owner via meta-tx if sig_type=2; deposit wallet meta-tx if sig_type=3). Gas: well under $0.01 in POL. The docs note many redemption paths are gas-subsidized via meta-tx so the bot may not need POL at all if it goes through the relayer.

Sample Python (untested skeleton, adapt for your sig_type):
```python
from web3 import Web3
w3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com"))
CTF = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"  # Polygon CTF
ctf  = w3.eth.contract(address=CTF, abi=CTF_ABI)
tx = ctf.functions.redeemPositions(
    Web3.to_checksum_address("0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"),  # pUSD
    "0x" + "00"*32,
    market["conditionId"],
    [1, 2],
).build_transaction({"from": funder, "nonce": w3.eth.get_transaction_count(funder),
                     "gas": 300_000})
```
Source: https://robottraders.io/blog/polymarket-auto-redeem-python

Polymarket's website redeems automatically when you log in; bots must batch their own. A daily cron is enough — there's no deadline.

---

## 6. SDKs and the Python ecosystem

### 6.1 What's official

| Package | Repo | Latest | Status |
|---|---|---|---|
| `py-clob-client` (v1) | https://github.com/Polymarket/py-clob-client | 0.34.6 (Feb 2026) | **Archived** but works |
| `py-clob-client-v2` | https://github.com/Polymarket/py-clob-client-v2 | 1.0.1 (May 2026) | Active |
| `py-order-utils` | https://github.com/Polymarket/python-order-utils | 0.3.2 (Jul 2024) | Pulled in transitively |
| `poly-py-eip712-structs` | https://github.com/Polymarket/poly-py-eip712-structs | n/a | Signing helpers |

V1 vs V2 differences:
- V2 supports `signature_type=3` (POLY_1271 deposit wallets).
- V2 EIP-712 order struct uses `timestamp` + `metadata` + `builder` fields instead of `expiration` + `nonce` + `feeRateBps` + `taker` — but the **payload sent on the wire** still accepts `expiration` for GTD compatibility.
- V2 adds builder-code attribution (volume rewards) — irrelevant for us.

For our project I'd start with **v1** (Docker image already builds it, dependency footprint is well-understood) and migrate to v2 only if we hit the `signature_type=3` requirement.

### 6.2 Coverage

`py-clob-client` covers:
- Reading prices, books, midpoints, trades, server time (read-only, no auth).
- L1/L2 auth bootstrap.
- Order creation, signing, posting, cancelling.
- User orders + trades fetch.
- Allowance setup is **NOT** included — see https://nautilustrader.io/docs/latest/integrations/polymarket/ for a separate `set_allowances.py` script.

It does **NOT** cover Gamma market discovery — you need plain `requests` against `gamma-api.polymarket.com` (which is fine — Gamma is unauthenticated REST).

### 6.3 Python version compatibility

- `py-clob-client` v0.34.x: officially Python ≥3.9. Tested by community on 3.11/3.12. No GitHub issues open for 3.14 specifically, but `eth-account` and `web3` (transitive deps) have had 3.12→3.13 friction in the past.
- The Docker image `python:3.12-slim` is the safest bet — match it locally and you avoid surprises. Python 3.14 on the host should still work because the orchestrator can run inside the container.

### 6.4 Dependency footprint of py-clob-client

Transitively (from setup.py / pyproject):
- `py-order-utils` (signing scaffolding)
- `poly-py-eip712-structs`
- `eth-account` (signs the EIP-712 payload)
- `web3` (for the eth-account base; not used directly for RPC calls)
- `httpx` or `requests` (HTTP)
- `pydantic` (model)

Add `eth-typing`, `eth-utils`, `eth-abi`, `parsimonious`, `protobuf` (transitively). Total install size ~30-40 MB.

For redemption you'll want **`web3`** at top level so you can speak to Polygon RPC directly (the SDK doesn't expose `redeemPositions`).

---

## 7. Testnet / sandbox

### 7.1 What exists

- Polygon **Amoy** testnet, chain_id `80002`, replaces Mumbai. (https://polygon.technology/blog/introducing-the-amoy-testnet-for-polygon-pos)
- CLOB on Amoy: there's a test CLOB at `https://clob-staging.polymarket.com` (referenced by community; not in current docs).
- Different contract addresses on Amoy (CTF Exchange + USDC). Hard-coded in the SDK behind `chain_id=80002`.

### 7.2 What doesn't exist on testnet

- **The 15-min crypto markets are mainnet-only.** No `btc-updown-15m-*` slugs exist on the testnet Gamma.
- Therefore end-to-end testing of "discover market → BUY → wait → redeem" cannot be done in sandbox.

### 7.3 Pragmatic alternative

Run on mainnet with the smallest possible bet:
- 5 shares at p=0.50 → $2.50 risked, fee ~5¢, max loss ~$2.55.
- Take three or four cycles to verify the wire format end-to-end, then turn the size up.

Polymarket's burst rate-limit (5,000 POST /order per 10s) means even rapid iteration is safe; the only cost is the actual USDC.

Faucets for Amoy MATIC/test-USDC (irrelevant for us since no 15m markets there):
- https://amoy.polygonscan.com/

---

## 8. Pitfalls and surprises — money-burners to know about

### 8.1 Direction mix-up (UP vs DOWN token_id)

The single most expensive bug. The `clobTokenIds` array is **always** in the same order as `outcomes`, but neither is guaranteed to be `[Up, Down]` (in our live sample it was, but new markets could in principle ship `[Down, Up]`). Code defensively:

```python
outcomes  = json.loads(market["outcomes"])
token_ids = json.loads(market["clobTokenIds"])
TOKEN_UP   = token_ids[outcomes.index("Up")]
TOKEN_DOWN = token_ids[outcomes.index("Down")]
assert TOKEN_UP != TOKEN_DOWN
```

Log both tokens at decision time so a post-mortem can prove which side was bought.

### 8.2 BUY-side vs SELL-side

`MarketOrderArgs(side=BUY, amount=X)` interprets `amount` as **USDC to spend**.
`MarketOrderArgs(side=SELL, amount=X)` interprets `amount` as **shares to sell**.

We always BUY → always pass `amount=usdc_dollars`. Adding a SELL path later means swapping units. The nautilustrader integration calls this out as the #1 user mistake.
(https://nautilustrader.io/docs/latest/integrations/polymarket/ — "Quantity Semantics Mismatch")

### 8.3 USDC decimals

USDC on Polygon has **6 decimals**. The `makerAmount`/`takerAmount` fields in the signed Order are uint256 in those base units. If your SDK shim ever computes them by hand, off-by-1e6 sends 1 millionth or a million times the intended amount. **Use the SDK** — `create_market_order` does this correctly.

### 8.4 `signatureType` mismatch with `funder`

If you connect MetaMask to polymarket.com once, the site auto-deploys a Gnosis Safe proxy for you. From that point onwards, your **funds live in the Safe, not your EOA**. If your bot uses `signature_type=0` (EOA) it will report no balance even though Polymarket shows $X — because that $X is in the proxy, not the EOA. Fix: `signature_type=2` + `funder=<safe_address>`.

Check the proxy address at https://polymarket.com/settings.

### 8.5 Fee rate signed into the order

The `feeRateBps` in the signed Order must equal the market's current taker base fee bps (here 1000). If Polymarket flips fees on a market (e.g. raises a sport's taker fee), pre-signed orders get rejected with a signature-mismatch error. Always fetch the rate via `client.get_fee_rate_bps(token_id)` right before signing.

### 8.6 Allowances (EOA / type 0 only)

If you use a raw EOA (`signature_type=0`), you must ERC-20 `approve` USDC and ERC-1155 `setApprovalForAll` on the CTF Exchange contract — *otherwise the first order fills off-chain and fails on-chain*, leaving you with a "submitted" but unsettled trade. The Gnosis Safe flow (`type=2`) sets this up for you when polymarket.com deploys the safe.

If you must use type 0: nautilustrader ships a standalone `set_allowances.py`. Run once.

### 8.7 Gas / POL requirements

- For `signature_type=0` (EOA): your EOA spends POL gas for every settlement transaction. Budget a couple cents per fill on Polygon (~$0.002 in good times, $0.05 worst case).
- For `signature_type=2` (Gnosis Safe via Polymarket relayer): Polymarket relays settlement gas in most cases (`gasless` flag). You still want a small POL balance (1-5 POL) for redemption transactions you call yourself.
- For `signature_type=3` (POLY_1271 deposit wallet): fully gasless via Polymarket's meta-tx infra.

Don't drain the EOA to zero — you'll lock yourself out of redemptions.

### 8.8 Geoblocking

`https://polymarket.com/api/geoblock` is called by the CLOB on every order. France, Germany, Belgium, Netherlands, Italy, Poland (close-only) are all blocked. The doc page (https://docs.polymarket.com/api-reference/geoblock) and help center (https://help.polymarket.com/en/articles/13364163-geographic-restrictions) list the full set.

- Hosting in Dublin, Ireland (eu-west-1) — Ireland is not on the blocklist, latency to eu-west-2 ~2 ms. This is the de-facto bot location.
- Hosting in France — order POSTs return `{"error":"GEO_BLOCKED"}`.
- VPNs are explicitly disallowed by ToS §2.1.4.

### 8.9 Resolution source ≠ training data

Our model is trained on Bitstamp 15m candles. Polymarket resolves on the **Chainlink BTC/USD aggregated price stream** at https://data.chain.link/streams/btc-usd. Between Bitstamp's print and Chainlink's median-of-many-feeds tick, you should expect a few dollars of disagreement on flat moves and the boundary `>=` rule makes a 1¢ Up flat resolve "Up". Concretely:

- A model that bets "Down" with high confidence on a flat candle will lose on Polymarket.
- A model trained with `next_close > next_open` (strict >) is mis-aligned with Polymarket's `>=`.

Consider retraining with the Polymarket rule, or at least pulling Chainlink's historical stream to backtest against the actual resolution price.

### 8.10 The market freezes ~5 s before close

`acceptingOrders` flips to false a few seconds before `endDate`. Posting after that returns `400 NOT_ACCEPTING_ORDERS`. Our bot already wakes at xx:00/15/30/45 and decides on the *prior* candle's close, so we have ~14 min 50 s to act — plenty of headroom. But: if the bot is slow to start (Docker cold start, network blip), the first cycle of the day may miss. Log `acceptingOrders` before every POST.

### 8.11 KYC / region

- No KYC on the international site (mainnet) for orders ≤ thresholds. Just IP gating.
- Polymarket US (separate platform, `docs.polymarket.us`) does require KYC.
- For a server in Dublin: no KYC, no identity verification.

### 8.12 The "spread" trap on illiquid windows

Our live capture showed `bestBid: 0.66 / bestAsk: 0.67` on a *resolved* market — i.e. ghosts of the last orders, not real liquidity. The market `closed: true`. Always check `market.acceptingOrders` AND `market.closed == false` before sending size. The cleanest rule: only trade markets where `active && acceptingOrders && !closed`.

---

## 9. Suggested wire-up for our orchestrator

A compact "shopping list" for `live/orchestrator.py` to add Polymarket execution on top of the model decision:

1. **Boot once**:
   - Resolve `signature_type` (`2` if `POLYMARKET_PROXY_ADDRESS` differs from EOA, else `3` if using deposit wallet).
   - `client = ClobClient(...)` + `set_api_creds(create_or_derive_api_creds())`.
   - Sanity: `client.get_ok()` then `client.get_server_time()` and compare to local clock.
   - Sanity: hit `https://polymarket.com/api/geoblock` from the host; if `blocked=true`, abort startup.
   - Cache `api_key/secret/passphrase` to a sidecar file (`.poly.creds.json`, 0600 perms) for restart.
2. **Each xx:00/15/30/45 UTC**:
   - Compute `slug = f"btc-updown-15m-{next_window_start_unix}"`.
   - `requests.get(f"https://gamma-api.polymarket.com/markets/slug/{slug}")`.
   - Verify `active && acceptingOrders && enableOrderBook && !closed`.
   - Parse `outcomes`, `clobTokenIds` → derive `TOKEN_UP`, `TOKEN_DOWN`.
   - Map model decision → `token_id`.
   - `client.get_order_book(token_id)` → compute slippage-capped `limit = min(best_ask + 0.02, 0.99)`.
   - Compute `amount_usdc = BET_FRACTION * free_balance` clipped to [`BET_MIN_USDC`, `BET_MAX_FRACTION * free`].
   - Reject if `amount_usdc / limit < orderMinSize` (i.e. resulting shares < 5).
   - `mo = MarketOrderArgs(token_id=..., amount=amount_usdc, side=BUY, order_type=OrderType.FAK, price=limit)`.
   - `signed = client.create_market_order(mo)`; `resp = client.post_order(signed, OrderType.FAK)`.
   - Append `resp` (success, status, takingAmount, transactionsHashes) to `live/logs/bets.csv` along with `condition_id`, `token_id`, `slug`, `decision`, `confidence`, `limit_price`, `amount_usdc`.
3. **Daily cron** (separate process):
   - Walk yesterday's `bets.csv`, for each unique `condition_id` call `redeemPositions(pUSD, 0x000…000, condition_id, [1,2])` via web3.
   - Log redemption tx hashes and final pUSD balance.

---

## 10. Open questions / things the docs are silent on

1. **`/price` BUY vs SELL semantics** — live response inverted my reading. Always cross-check with `/book` before relying on `/price`.
2. **Exact behavior of `automaticallyActive` flag** — when is a window opened for trading vs. when can the bot pre-position? Empirically the slug is queryable ~10-15 min before its window starts (we saw market `1778779800` available at `1778778900`-ish window-2 from current).
3. **`feeSchedule` vs `takerBaseFee:1000`** — the JSON ships both. The Order's `feeRateBps` should be `1000` (basis-point base rate), then the contract applies the `p*(1-p)` curve. Confirm with one tiny live order whose fees you observe in the trade event.
4. **Whether `POLY_1271` (sig_type=3) can sign orders from a pure server with no Polymarket website history.** Docs imply yes ("Recommended for new API users"); community is sparse on this. Safer first deployment is sig_type=2 with an existing safe.
5. **Maker rebates on 15-min markets** — `makerRebatesFeeShareBps: 10000`, `rewards.min_size: 50`. Resting GTC would earn rebates, but we are deliberately a taker (FAK) so this doesn't apply to v1 of the bot.

---

## 11. URL appendix (everything cited)

Official documentation:
- https://docs.polymarket.com/
- https://docs.polymarket.com/llms.txt
- https://docs.polymarket.com/api-reference/introduction
- https://docs.polymarket.com/api-reference/rate-limits
- https://docs.polymarket.com/api-reference/geoblock
- https://docs.polymarket.com/api-reference/clients-sdks
- https://docs.polymarket.com/api-reference/markets/get-market-by-slug
- https://docs.polymarket.com/concepts/markets-events
- https://docs.polymarket.com/developers/CLOB/authentication
- https://docs.polymarket.com/developers/CLOB/clients/methods-l1
- https://docs.polymarket.com/developers/CLOB/clients/methods-l2
- https://docs.polymarket.com/developers/CLOB/orders/orders
- https://docs.polymarket.com/developers/gamma-markets-api/overview
- https://docs.polymarket.com/developers/proxy-wallet
- https://docs.polymarket.com/developers/market-makers/maker-rebates-program
- https://docs.polymarket.com/quickstart
- https://docs.polymarket.com/quickstart/fetching-data
- https://docs.polymarket.com/trading/ctf/redeem
- https://help.polymarket.com/en/articles/13364163-geographic-restrictions

Code / SDK / contracts:
- https://github.com/Polymarket/py-clob-client
- https://github.com/Polymarket/py-clob-client-v2
- https://github.com/Polymarket/python-order-utils
- https://github.com/Polymarket/poly-py-eip712-structs
- https://github.com/Polymarket/clob-client-v2
- https://github.com/Polymarket/rs-clob-client
- https://github.com/Polymarket/ctf-exchange
- https://github.com/Polymarket/neg-risk-ctf-adapter
- https://github.com/Polymarket/agents/blob/main/agents/polymarket/gamma.py
- https://github.com/Polymarket/py-clob-client/issues/121 (FOK decimal error)
- https://github.com/Polymarket/py-clob-client/issues/187 (401 / invalid key)
- https://github.com/Polymarket/py-clob-client/issues/244 (timestamp formula)
- https://github.com/Polymarket/py-clob-client/issues/288 (400 invalid signature EOA)
- https://github.com/Polymarket/py-clob-client/issues/344 (sig_type=3 not in v1)
- https://pypi.org/project/py-clob-client/
- https://pypi.org/project/py-clob-client-v2/
- https://polygonscan.com/address/0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e (CTF Exchange)

Live data + product:
- https://polymarket.com/crypto/15M
- https://polymarket.com/event/btc-updown-15m-1778779800
- https://gamma-api.polymarket.com/markets/slug/btc-updown-15m-1778779800
- https://clob.polymarket.com/book?token_id=…
- https://data.chain.link/streams/btc-usd

Third-party but useful:
- https://nautilustrader.io/docs/latest/integrations/polymarket/ — by far the densest practitioner write-up
- https://agentbets.ai/guides/py-clob-client-reference/
- https://agentbets.ai/guides/py-clob-client-create-order/
- https://agentbets.ai/guides/polymarket-auth-troubleshooting/
- https://github.com/handiko/Polymarket-Market-Finder
- https://github.com/FrondEnt/PolymarketBTC15mAssistant
- https://chainstack.com/polymarket-api-for-developers/
- https://quantjourney.substack.com/p/understanding-the-polymarket-fee
- https://robottraders.io/blog/polymarket-auto-redeem-python
- https://www.cryptopolitan.com/polymarkets-15-minute-up-down/
- https://medium.com/@utommy.abs/polymarket-15-minute-dump-and-hedge-bot-89018d7aa8fb
