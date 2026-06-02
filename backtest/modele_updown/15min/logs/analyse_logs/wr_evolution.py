"""Interactive win-rate evolution chart for the team backtest.

Reads 15min/logs/team_taken.csv (one row per bet the team placed, with a
`result` column = win|loss plus the bet bar's OHLC) and renders a dark,
TradingView-style HTML chart showing how the team win rate evolved.

The headline curve is the cumulative win rate. Its thin-sample head (the
first bets, where one win/loss swings it to 0/100 %) is shrunk toward
50 % by a light Bayesian prior; past the first weeks the prior washes out
and the curve is the true cumulative WR. A rolling-WR curve (window
editable in days, warmed up with pre-test bets) shows recent form.

Layout, TradingView-style:
  - the main pane holds every win-rate curve (they share the % scale)
  - any series with a DIFFERENT unit opens in its own pane BELOW the main
    chart (BTC price, ATR, compound capital) -- never overlaid, scales differ
  - one shared time axis, one full-height crosshair across every pane

What you get:
  - dark theme, X+Y zoom in sync (custom wheel handler), drag-to-pan
  - a collapsible control panel (top-right): toggle each series, read its
    value on hover, set the rolling window in days
  - a compound-capital curve: $500 start, a TIERED stake (10 % under $1k,
    then 5/2/1 % past $1k/$10k/$50k, 3 % in $100k-$1M, 1 % past $1M),
    x2 on a win, total loss on a miss. From $100k of total capital each win
    is split 50/50 -- half reinvested, half set aside (own log-scale pane)
  - trend-line drawing tools, PNG export

Output: wr_evolution.html next to this script (self-contained, offline).

Run (from anywhere):
    python wr_evolution.py [path/to/team_taken.csv]
"""
from __future__ import annotations

import json
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

# Make accented console output safe on a Windows cp1252 terminal.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ------------------------------------------------------------------ tunables
DEFAULT_ROLLING_DAYS = 120  # rolling-WR window, in days (live-editable in UI)
MAX_WARMUP_DAYS = 365       # pre-test bets pulled in so the rolling WR is full
HEAD_PRIOR = 8              # Bayesian prior taming the cumulative-WR head only
ATR_PERIOD = 14             # bars in the ATR range average (then 0-100 scaled)
ATR_MA_DAYS = 120           # default ATR rolling-average window, days (UI-editable)
CONFIDENCE_Z = 1.96         # 95 % Wilson score interval
CAPITAL_START = 500.0       # starting balance of the simulated capital curve
# Tiered stake: the fraction of the at-risk balance staked on each bet steps
# DOWN as the total capital grows. CAPITAL_TIERS is scanned top first;
# under the first threshold CAPITAL_FRACTION_BASE applies. The rate is
# re-read before every bet, so crossing a tier (up OR down) switches it.
CAPITAL_FRACTION_BASE = 0.10               # capital < $1k     -> 10 %
CAPITAL_TIERS = (
    (1_000_000, 0.01),                     # capital >= $1M    -> 1 %
    (100_000,   0.03),                     # capital >= $100k  -> 3 %
    (50_000,    0.01),                     # capital >= $50k   -> 1 %
    (10_000,    0.02),                     # capital >= $10k   -> 2 %
    (1_000,     0.05),                     # capital >= $1k    -> 5 %
)
# Profit protection: once the TOTAL capital reaches this level, each winning
# bet's profit is split -- half reinvested into the at-risk balance, half
# moved to a set-aside pot that is never staked again (but still counts in
# the displayed total). The stake fraction follows CAPITAL_TIERS above
# (3 % in the $100k-$1M band, 1 % past $1M).
CAPITAL_PROFIT_SPLIT_FROM = 100_000.0      # >= $100k -> split every win
CAPITAL_PROFIT_REINVEST   = 0.50           # 50 % reinvested, 50 % set aside

_DAY_MS = 86_400_000

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "team_taken.csv"
OUTPUT_HTML = Path(__file__).resolve().parent / "wr_evolution.html"

# TradingView-ish dark palette.
C = dict(
    bg="#131722",
    grid="#222631",
    axis="#2a2e39",
    text="#d1d4dc",
    dim="#787b86",
    smoothed="#3b9aff",
    edge_fill="rgba(38,166,154,0.10)",
    rolling="rgba(255,183,77,0.5)",
    raw="#5d6b8a",
    baseline="#787b86",
    final="#26a69a",
    band="rgba(59,154,255,0.14)",
    spike="#9598a1",
    price="#4dd0e1",
    atr="#ba68c8",
    capital="#66bb6a",
    bets="#ef5350",
)

# Injected after the plot is built. Jobs:
#  1. custom wheel zoom -- X and every active Y axis move in ONE relayout,
#  2. the collapsible control panel that replaces the legend,
#  3. live recompute of the rolling WR when the day count changes,
#  4. open / close the price, ATR & capital panes below the main chart,
#  5. a full-height crosshair that spans every pane.
# The data arrays (WR_WINS / WR_TS / DEFAULT_DAYS) are prepended in write_chart.
_AUTO_Y_JS = """
(function () {
  var gd = document.getElementById("{plot_id}");
  if (!gd || !gd.on) { return; }

  // ===== trace indices (must match build_figure add_trace order) ==========
  var T_CI = 0, T_BASE = 1, T_SMOOTH = 2, T_ROLL = 3,
      T_PRICE = 4, T_ATR = 5, T_CAP = 6, T_BETS = 7;

  // Lower panes: each indicator is simply on or off. "on" => its own pane.
  var state = { price: false, atr: false, cap: false, bets: false };
  // WR axis scale: false = fixed 50-70 %, true = auto-fit (min..max).
  var wrAuto = false;
  // Read-out persistence: a redraw fires a spurious unhover, so we keep the
  // box up until `scrollUntil` (refreshed on every wheel / pan event).
  var hideTimer = null, scrollUntil = 0;

  // Per-trace Y data, by trace index. The source of truth for the read-out
  // box and the auto-fit -- never gd.data[i].y (Plotly base64-packs it).
  var SDATA = {};
  SDATA[T_SMOOTH] = WR_CUM;
  SDATA[T_ROLL]   = rollingWR(DEFAULT_DAYS);
  SDATA[T_PRICE]  = WR_PRICE;
  SDATA[T_ATR]    = rollingATR(ATR_DAYS);
  SDATA[T_CAP]    = WR_CAPITAL;
  SDATA[T_BETS]   = WR_BETS;

  // ===== shared helpers ==================================================
  var xcache = null;
  var xmin = Infinity, xmax = -Infinity;
  var selfRelayout = false;

  function toMs(v) { return (typeof v === "number") ? v : new Date(v).getTime(); }

  function buildCache() {
    xmin = Infinity; xmax = -Infinity;
    xcache = (gd.data || []).map(function (tr) {
      var xs = tr.x || [];
      var out = new Float64Array(xs.length);
      for (var i = 0; i < xs.length; i++) {
        var m = toMs(xs[i]); out[i] = m;
        if (m < xmin) { xmin = m; }
        if (m > xmax) { xmax = m; }
      }
      return out;
    });
  }
  try { buildCache(); } catch (e) { xcache = null; }

  function rangeFor(x0, x1, indices) {
    var data = gd.data || [];
    if (!xcache || xcache.length !== data.length) { buildCache(); }
    if (x1 < x0) { var sw = x0; x0 = x1; x1 = sw; }
    var lo = Infinity, hi = -Infinity;
    for (var j = 0; j < indices.length; j++) {
      var ti = indices[j], tr = data[ti];
      if (!tr || tr.visible === "legendonly" || tr.visible === false) { continue; }
      var xs = xcache[ti], ys = SDATA[ti];
      if (!xs || !ys) { continue; }
      for (var i = 0; i < xs.length; i++) {
        var tx = xs[i];
        if (tx >= x0 && tx <= x1) {
          var v = ys[i];
          if (v === null || v === undefined || isNaN(v)) { continue; }
          if (v < lo) { lo = v; }
          if (v > hi) { hi = v; }
        }
      }
    }
    if (lo === Infinity) { return null; }
    return [lo, hi];               // raw extent; yUpdates pads (linear / log)
  }

  // Auto-fit groups. The WR axis joins only when wrAuto is on (the top-left
  // toggle); otherwise it stays locked to the fixed 50-70 % band. The 50 %
  // baseline is excluded so "auto" spans the curves, not the flat 0.5 line.
  function activeGroups() {
    var g = [];
    if (wrAuto) {
      g.push({ key: "yaxis", traces: [T_SMOOTH, T_ROLL, T_CI] });
    }
    if (state.price) { g.push({ key: "yaxis2", traces: [T_PRICE] }); }
    // yaxis3 (ATR) is a fixed 0-100 index axis -- no auto-fit on zoom.
    if (state.cap)   { g.push({ key: "yaxis4", traces: [T_CAP], log: true }); }
    if (state.bets)  { g.push({ key: "yaxis5", traces: [T_BETS] }); }
    return g;
  }

  // Pad each group's raw extent by 12 %. The capital pane is a log axis, so
  // its range must be in log10 units -- convert the extent before padding.
  function yUpdates(x0, x1) {
    var upd = {}, groups = activeGroups();
    for (var i = 0; i < groups.length; i++) {
      var grp = groups[i];
      var r = rangeFor(x0, x1, grp.traces);
      if (!r) { continue; }
      var lo = r[0], hi = r[1];
      if (grp.log) {
        lo = Math.log10(lo > 0 ? lo : 1e-9);
        hi = Math.log10(hi > 0 ? hi : 1e-9);
      }
      var pad = (hi - lo) * 0.12;
      if (!(pad > 0)) { pad = Math.abs(hi || lo) * 0.02 || 0.02; }
      upd[grp.key + ".range"] = [lo - pad, hi + pad];
    }
    return upd;
  }

  function doRelayout(upd) {
    selfRelayout = true;
    var p = Plotly.relayout(gd, upd);
    if (p && typeof p.then === "function") {
      p.then(function () { selfRelayout = false; },
             function () { selfRelayout = false; });
    } else { selfRelayout = false; }
  }

  // ===== custom wheel zoom: X + every active Y axis in ONE relayout =======
  var ZOOM_IN = 0.88, ZOOM_OUT = 1 / 0.88;
  var pendFactor = 1, pendPivot = 0, wheelRAF = false;

  gd.addEventListener("wheel", function (evt) {
    if (panel && panel.contains(evt.target)) { return; }
    if (scaleBtn && scaleBtn.contains(evt.target)) { return; }
    var fl = gd._fullLayout;
    if (!fl || !fl.xaxis) { return; }
    var xa = fl.xaxis;
    var rect = gd.getBoundingClientRect();
    var px = evt.clientX - rect.left;
    if (px < xa._offset || px > xa._offset + xa._length) { return; }
    evt.preventDefault();
    scrollUntil = Date.now() + 280;          // keep the read-out box alive
    pendFactor *= (evt.deltaY > 0) ? ZOOM_OUT : ZOOM_IN;
    pendPivot = xa.p2l(px);
    if (wheelRAF) { return; }
    wheelRAF = true;
    requestAnimationFrame(applyWheelZoom);
  }, { passive: false });

  function applyWheelZoom() {
    wheelRAF = false;
    var fl = gd._fullLayout;
    if (!fl || !fl.xaxis) { pendFactor = 1; return; }
    var xa = fl.xaxis;
    var f = pendFactor; pendFactor = 1;
    var r0 = xa.r2l(xa.range[0]), r1 = xa.r2l(xa.range[1]);
    var nr0 = pendPivot - (pendPivot - r0) * f;
    var nr1 = pendPivot + (r1 - pendPivot) * f;
    if (nr0 < xmin) { nr0 = xmin; }
    if (nr1 > xmax) { nr1 = xmax; }
    if (nr1 - nr0 < 60000) { return; }
    var upd = yUpdates(nr0, nr1);
    upd["xaxis.range"] = [nr0, nr1];          // xaxis2/3 follow via matches:"x"
    doRelayout(upd);
  }

  // ===== pan / range buttons / restyle: Y follows X ======================
  var scheduled = false;
  function rescaleY() {
    var fl = gd._fullLayout;
    if (!fl || !fl.xaxis || !fl.xaxis.range) { return; }
    var upd = yUpdates(toMs(fl.xaxis.range[0]), toMs(fl.xaxis.range[1]));
    for (var k in upd) { doRelayout(upd); return; }
  }
  function schedule() {
    if (scheduled) { return; }
    scheduled = true;
    requestAnimationFrame(function () { scheduled = false; rescaleY(); });
  }
  gd.on("plotly_relayout", function (ed) {
    if (selfRelayout) { return; }
    for (var k in ed) {
      if (k.indexOf("xaxis") === 0) { schedule(); return; }
    }
  });
  gd.on("plotly_restyle", function () { if (!selfRelayout) { schedule(); } });
  // Interactive pan also keeps the read-out box alive.
  gd.on("plotly_relayouting", function () { scrollUntil = Date.now() + 280; });

  // ===== hide Plotly's native hover visuals (the panel shows values) =====
  var styleEl = document.createElement("style");
  styleEl.textContent = [
    ".hoverlayer .hovertext  { display:none !important; }",
    ".hoverlayer text         { display:none !important; }",
    ".hoverlayer rect         { display:none !important; }",
    ".hoverlayer .hoverpoint  { display:none !important; }",
    ".hoverlayer path         { display:none !important; }",
  ].join(" ");
  document.head.appendChild(styleEl);

  // ===== rolling WR recompute (sliding time window, in days) =============
  function rollingWR(days) {
    // Slide over warmup + test bets, then return only the test slice. The
    // warmup gives every test bet a full window right from the first one.
    var win = days * 86400000, m = ROLL_WINS.length, out = new Array(m);
    var lo = 0, acc = 0;
    for (var i = 0; i < m; i++) {
      acc += ROLL_WINS[i];
      while (ROLL_TS[i] - ROLL_TS[lo] > win) { acc -= ROLL_WINS[lo]; lo++; }
      out[i] = (ROLL_TS[i] - ROLL_TS[0] >= win) ? (acc / (i - lo + 1)) : null;
    }
    return out.slice(TEST_OFFSET);
  }

  // ===== ATR rolling average (sliding time window, in days) ==============
  function rollingATR(days) {
    // Sliding time-window mean of the raw 0-100 ATR over the test bets.
    // Partial windows early -> the line is full from the first valid bet.
    var win = days * 86400000, m = WR_ATR.length, out = new Array(m);
    var lo = 0, acc = 0, cnt = 0;
    for (var i = 0; i < m; i++) {
      var v = WR_ATR[i];
      if (v === v) { acc += v; cnt++; }            // v === v : not NaN
      while (WR_TS[i] - WR_TS[lo] > win) {
        var lv = WR_ATR[lo];
        if (lv === lv) { acc -= lv; cnt--; }
        lo++;
      }
      out[i] = (cnt > 0) ? (acc / cnt) : null;
    }
    return out;
  }

  // ===== open / close the lower panes ===================================
  // No overlay mode: price and ATR each get their own pane, stacked below
  // the main win-rate chart, because their unit ($) differs from the WR %.
  // Domain split per visible-pane count: main on top, indicator panes
  // stacked below it. Keys 1..3 -> up to three open panes (price/atr/cap).
  var DOMAINS = {
    1: { main: [0.40, 1.0], panes: [[0.0, 0.36]] },
    2: { main: [0.56, 1.0], panes: [[0.30, 0.52], [0.0, 0.26]] },
    3: { main: [0.55, 1.0], panes: [[0.37, 0.51], [0.19, 0.33], [0.0, 0.15]] },
    4: { main: [0.62, 1.0], panes: [[0.465, 0.585], [0.31, 0.43],
                                    [0.155, 0.275], [0.0, 0.12]] },
  };

  function paneLayout() {
    var order = [];
    if (state.price) { order.push("price"); }
    if (state.atr)   { order.push("atr"); }
    if (state.cap)   { order.push("cap"); }
    if (state.bets)  { order.push("bets"); }
    var P = order.length, lay = {};
    var YA = { price: "yaxis2", atr: "yaxis3", cap: "yaxis4", bets: "yaxis5" };
    var XA = { price: "xaxis2", atr: "xaxis3", cap: "xaxis4", bets: "xaxis5" };

    // Default: main fills everything; pane axes collapse to a full overlap.
    // No trace sits on a pane axis while its pane is closed (render() parks
    // price / atr / capital / bets back on the main axes), so it is hidden.
    lay["yaxis.domain"]  = [0.0, 1.0];
    lay["yaxis2.domain"] = [0.0, 1.0];
    lay["yaxis3.domain"] = [0.0, 1.0];
    lay["yaxis4.domain"] = [0.0, 1.0];
    lay["yaxis5.domain"] = [0.0, 1.0];
    if (P > 0) {
      var dom = DOMAINS[P];
      lay["yaxis.domain"] = dom.main;
      for (var k = 0; k < P; k++) {
        lay[YA[order[k]] + ".domain"] = dom.panes[k];
      }
    }

    lay["xaxis2.visible"] = state.price;
    lay["yaxis2.visible"] = state.price;
    lay["xaxis3.visible"] = state.atr;
    lay["yaxis3.visible"] = state.atr;
    lay["xaxis4.visible"] = state.cap;
    lay["yaxis4.visible"] = state.cap;
    lay["xaxis5.visible"] = state.bets;
    lay["yaxis5.visible"] = state.bets;

    // tick labels only on the bottom-most visible pane.
    var bottom = P ? XA[order[P - 1]] : "xaxis";
    lay["xaxis.showticklabels"]  = (bottom === "xaxis");
    lay["xaxis2.showticklabels"] = (bottom === "xaxis2");
    lay["xaxis3.showticklabels"] = (bottom === "xaxis3");
    lay["xaxis4.showticklabels"] = (bottom === "xaxis4");
    lay["xaxis5.showticklabels"] = (bottom === "xaxis5");

    // pane name tags (annotations: 0 = WR-final hline, 1 = price, 2 = atr,
    // 3 = capital, 4 = bets/week).
    lay["annotations[1].visible"] = state.price;
    lay["annotations[2].visible"] = state.atr;
    lay["annotations[3].visible"] = state.cap;
    lay["annotations[4].visible"] = state.bets;
    return lay;
  }

  function render() {
    // Move each indicator onto its pane axes when on, park it back on the
    // main axes (where it draws nothing while legendonly) when off. Parking
    // keeps the closed-state layout identical to a plain single chart.
    Plotly.update(gd,
      { visible: [state.price ? true : "legendonly",
                  state.atr ? true : "legendonly",
                  state.cap ? true : "legendonly",
                  state.bets ? true : "legendonly"],
        xaxis:   [state.price ? "x2" : "x", state.atr ? "x3" : "x",
                  state.cap ? "x4" : "x", state.bets ? "x5" : "x"],
        yaxis:   [state.price ? "y2" : "y", state.atr ? "y3" : "y",
                  state.cap ? "y4" : "y", state.bets ? "y5" : "y"] },
      paneLayout(), [T_PRICE, T_ATR, T_CAP, T_BETS]);
    schedule();
  }

  // ===== full-height crosshair (spans every pane) ========================
  var vline = document.createElement("div");
  vline.style.cssText = "position:absolute;width:0;display:none;z-index:60;"
    + "border-left:1px dashed rgba(149,152,134,0.65);pointer-events:none;";
  gd.appendChild(vline);
  function showVline(dataX) {
    var fl = gd._fullLayout;
    if (!fl || !fl.xaxis) { return; }
    var xa = fl.xaxis;
    var r0 = xa.r2l(xa.range[0]), r1 = xa.r2l(xa.range[1]);
    var frac = (toMs(dataX) - r0) / (r1 - r0);
    vline.style.left = (xa._offset + frac * xa._length) + "px";
    vline.style.top = fl.margin.t + "px";
    vline.style.height = ((fl.height || 0) - fl.margin.t - fl.margin.b) + "px";
    vline.style.display = "block";
  }

  // ===== control panel (collapsible) =====================================
  var SERIES = [
    { idx: T_SMOOTH, name: "WR cumulatif",   color: "#3b9aff", fmt: "pct",
      kind: "wr" },
    { idx: T_ROLL,   name: "WR glissant",    color: "#ffb74d", fmt: "pct",
      kind: "wr", input: { days: DEFAULT_DAYS, recompute: rollingWR } },
    { idx: T_CI,     name: "Intervalle 95%", color: "#3b9aff", fmt: "none",
      kind: "wr" },
    { idx: T_PRICE,  name: "Prix BTC",       color: "#4dd0e1", fmt: "usd",
      kind: "ind", ind: "price" },
    { idx: T_ATR,    name: "ATR glissant",   color: "#ba68c8", fmt: "idx",
      kind: "ind", ind: "atr",
      input: { days: ATR_DAYS, recompute: rollingATR } },
    { idx: T_CAP,    name: "Capital (500 $, paliers)", color: "#66bb6a",
      fmt: "cap", kind: "ind", ind: "cap" },
    { idx: T_BETS,   name: "Paris / semaine",  color: "#ef5350", fmt: "count",
      kind: "ind", ind: "bets" },
  ];

  function isVisible(idx) {
    var v = gd.data[idx].visible;
    return v === true || v === undefined;
  }
  function isOn(s) {
    return (s.kind === "wr") ? isVisible(s.idx) : state[s.ind];
  }
  function fmtValue(fmt, v) {
    if (fmt === "pct") { return (v * 100).toFixed(2) + " %"; }
    if (fmt === "usd") { return "$" + Math.round(v).toLocaleString("en-US"); }
    if (fmt === "idx") { return v.toFixed(1); }
    if (fmt === "count") { return Math.round(v) + " paris"; }
    if (fmt === "cap") {
      if (v >= 1e7 || (v > 0 && v < 0.01)) { return "$" + v.toExponential(2); }
      return "$" + v.toLocaleString("en-US",
        { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }
    return "";
  }

  var panel = document.createElement("div");
  panel.style.cssText = [
    "position:absolute", "top:112px", "right:14px", "z-index:100",
    "background:rgba(19,23,34,0.96)", "border:1px solid #2a2e39",
    "border-radius:6px", "overflow:hidden",
    "font:11px/1.5 'Segoe UI',Roboto,sans-serif", "color:#d1d4dc",
    "box-shadow:0 6px 18px rgba(0,0,0,0.45)",
  ].join(";");
  gd.style.position = "relative";
  gd.appendChild(panel);

  var open = false;
  var header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;gap:8px;"
    + "padding:7px 11px;cursor:pointer;white-space:nowrap;font-weight:600;";
  var hIcon = document.createElement("span");
  hIcon.textContent = "\\u2630"; hIcon.style.color = "#9598a1";
  var hText = document.createElement("span");
  hText.textContent = "Series"; hText.style.flex = "1";
  var hChev = document.createElement("span"); hChev.style.color = "#9598a1";
  header.appendChild(hIcon); header.appendChild(hText); header.appendChild(hChev);
  panel.appendChild(header);

  var body = document.createElement("div");
  body.style.cssText = "display:none;padding:5px 11px 9px;"
    + "border-top:1px solid #2a2e39;min-width:170px;user-select:none;";
  panel.appendChild(body);

  function syncOpen() {
    body.style.display = open ? "block" : "none";
    hChev.textContent = open ? "\\u25b4" : "\\u25be";
  }
  header.onclick = function () { open = !open; syncOpen(); };
  syncOpen();

  var rows = {};
  function syncRow(s) {
    var on = isOn(s), r = rows[s.idx];
    r.box.style.background = on ? s.color : "transparent";
    r.rowEl.style.opacity = on ? "1" : "0.45";
  }
  function syncPanel() { SERIES.forEach(syncRow); }

  function toggleSeries(s) {
    if (s.kind === "wr") {
      Plotly.restyle(gd, { visible: isVisible(s.idx) ? "legendonly" : true },
                     [s.idx]);
    } else {
      state[s.ind] = !state[s.ind];
      render();
    }
    syncPanel();
  }

  SERIES.forEach(function (s) {
    var rowEl = document.createElement("div");
    rowEl.style.cssText = "display:flex;align-items:center;gap:7px;"
      + "padding:3px 3px;cursor:pointer;border-radius:3px;";
    rowEl.onmouseenter = function () { rowEl.style.background = "#1c2030"; };
    rowEl.onmouseleave = function () { rowEl.style.background = ""; };

    var box = document.createElement("span");
    box.style.cssText = "width:11px;height:11px;border-radius:2px;flex:none;"
      + "box-sizing:border-box;border:1.5px solid " + s.color + ";";
    var dot = document.createElement("span");
    dot.style.cssText = "width:14px;height:3px;border-radius:1px;flex:none;"
      + "background:" + s.color + ";";
    var nameEl = document.createElement("span");
    nameEl.textContent = s.name;
    nameEl.style.cssText = "flex:1;white-space:nowrap;padding-right:14px;";

    rowEl.appendChild(box); rowEl.appendChild(dot);
    rowEl.appendChild(nameEl);
    rowEl.onclick = function () { toggleSeries(s); };
    body.appendChild(rowEl);
    rows[s.idx] = { box: box, nameEl: nameEl, rowEl: rowEl };

    if (s.input) {
      var sub = document.createElement("div");
      sub.style.cssText = "display:flex;align-items:center;gap:5px;"
        + "padding:0 3px 4px 25px;color:#9598a1;font-size:10px;";
      var lbl = document.createElement("span"); lbl.textContent = "fenetre";
      var inp = document.createElement("input");
      inp.type = "number"; inp.min = "1"; inp.max = "365";
      inp.value = String(s.input.days);
      inp.style.cssText = "width:46px;background:#0e1117;color:#d1d4dc;"
        + "border:1px solid #2a2e39;border-radius:3px;padding:2px 4px;"
        + "font:10px 'Segoe UI',sans-serif;text-align:center;";
      inp.onclick = function (e) { e.stopPropagation(); };
      inp.onchange = function () {
        var d = parseInt(inp.value, 10);
        if (!(d >= 1)) { d = 1; }
        if (d > 365) { d = 365; }
        inp.value = String(d);
        var arr = s.input.recompute(d);        // rollingWR or rollingATR
        SDATA[s.idx] = arr;                    // keep the read-out in sync
        Plotly.restyle(gd, { y: [arr] }, [s.idx]);
      };
      var unit = document.createElement("span"); unit.textContent = "jours";
      sub.appendChild(lbl); sub.appendChild(inp); sub.appendChild(unit);
      body.appendChild(sub);
    }
  });
  syncPanel();

  // ===== WR scale toggle (top-left) ======================================
  // Switches the WR axis between the fixed 50-70 % band and an auto-fit that
  // tracks the curves' min..max in the visible window (a more volatile view).
  var scaleBtn = document.createElement("div");
  scaleBtn.style.cssText = [
    "position:absolute", "top:112px", "left:72px", "z-index:100",
    "background:rgba(19,23,34,0.96)", "border:1px solid #2a2e39",
    "border-radius:5px", "padding:5px 10px", "cursor:pointer",
    "font:11px 'Segoe UI',Roboto,sans-serif", "color:#d1d4dc",
    "white-space:nowrap", "box-shadow:0 4px 14px rgba(0,0,0,0.4)",
  ].join(";");
  gd.appendChild(scaleBtn);
  function syncScaleBtn() {
    scaleBtn.textContent = "\\u2195 Echelle WR : "
      + (wrAuto ? "auto (min-max)" : "50-70 %");
  }
  syncScaleBtn();
  scaleBtn.onclick = function () {
    wrAuto = !wrAuto;
    syncScaleBtn();
    if (wrAuto) { schedule(); }                       // auto-fit kicks in
    else { doRelayout({ "yaxis.range": [0.50, 0.70] }); }
  };

  // ===== hover read-out box (pinned right, follows the blue WR line) ======
  var hoverBox = document.createElement("div");
  hoverBox.style.cssText = [
    "position:absolute", "display:none", "z-index:110",
    "background:rgba(19,23,34,0.97)", "border:1px solid #2a2e39",
    "border-radius:5px", "padding:6px 9px", "pointer-events:none",
    "font:10.5px/1.55 'Segoe UI',Roboto,sans-serif", "color:#d1d4dc",
    "white-space:nowrap", "box-shadow:0 4px 14px rgba(0,0,0,0.5)",
  ].join(";");
  gd.appendChild(hoverBox);

  var MONTHS = ["jan","fev","mar","avr","mai","jun",
                "jul","aou","sep","oct","nov","dec"];
  var DAYW = ["Dim","Lun","Mar","Mer","Jeu","Ven","Sam"];
  function p2(n) { return n < 10 ? "0" + n : "" + n; }
  function fmtDate(ms) {
    var d = new Date(ms);
    return DAYW[d.getUTCDay()] + " " + p2(d.getUTCDate()) + " "
      + MONTHS[d.getUTCMonth()] + " " + d.getUTCFullYear()
      + " \\u00b7 " + p2(d.getUTCHours()) + "h" + p2(d.getUTCMinutes());
  }
  function nearestIdx(xMs) {
    var n = WR_TS.length;
    if (n === 0) { return 0; }
    if (xMs <= WR_TS[0]) { return 0; }
    if (xMs >= WR_TS[n - 1]) { return n - 1; }
    var lo = 0, hi = n - 1;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (WR_TS[mid] < xMs) { lo = mid + 1; } else { hi = mid; }
    }
    if (lo > 0 && (xMs - WR_TS[lo - 1]) < (WR_TS[lo] - xMs)) { return lo - 1; }
    return lo;
  }
  // One read-out line: coloured marker + series name + bold value. Built
  // with DOM nodes (no innerHTML) -- every field is our own trusted data.
  function readoutLine(color, name, valueText) {
    var line = document.createElement("div");
    var mk = document.createElement("span");
    mk.textContent = "\\u25aa";
    mk.style.cssText = "color:" + color + ";margin-right:5px";
    var nm = document.createElement("span");
    nm.textContent = name + " ";
    var vb = document.createElement("b");
    vb.style.color = "#fff";
    vb.textContent = valueText;
    line.appendChild(mk); line.appendChild(nm); line.appendChild(vb);
    return line;
  }

  function hideReadout() {
    hoverBox.style.display = "none";
    vline.style.display = "none";
  }

  gd.on("plotly_hover", function (ev) {
    if (!ev || !ev.points || !ev.points.length) { return; }
    var fl = gd._fullLayout;
    if (!fl || !fl.xaxis || !fl.yaxis) { return; }
    clearTimeout(hideTimer);                 // a hover cancels a pending hide
    var xv = ev.points[0].x;
    var idx = nearestIdx(toMs(xv));

    while (hoverBox.firstChild) { hoverBox.removeChild(hoverBox.firstChild); }
    var dEl = document.createElement("div");
    dEl.style.cssText = "color:#9598a1;margin-bottom:4px";
    dEl.textContent = fmtDate(WR_TS[idx]);
    hoverBox.appendChild(dEl);
    SERIES.forEach(function (s) {
      if (s.fmt === "none" || !isOn(s)) { return; }
      var arr = SDATA[s.idx];
      var y = arr ? arr[idx] : null;
      if (typeof y !== "number" || isNaN(y)) { return; }
      hoverBox.appendChild(readoutLine(s.color, s.name, fmtValue(s.fmt, y)));
    });
    hoverBox.style.display = "block";

    // Pinned to the right edge via CSS `right` (no width measurement, so it
    // can never slip off-screen). Vertical centre follows the blue WR-lisse
    // line, placed by offset + fraction*length.
    var blueY = SDATA[T_SMOOTH][idx];
    var ya = fl.yaxis, xa = fl.xaxis;
    var yr = ya.range || [0.5, 0.7];
    var yfrac = (blueY - yr[0]) / (yr[1] - yr[0]);
    var pixY = ya._offset + (1 - yfrac) * ya._length;
    if (!isFinite(pixY)) { pixY = ya._offset + ya._length / 2; }
    var bh = hoverBox.offsetHeight;
    var top = pixY - bh / 2;
    var pTop = ya._offset, pBot = ya._offset + ya._length;
    if (top < pTop) { top = pTop; }
    if (top + bh > pBot) { top = pBot - bh; }
    hoverBox.style.top = top + "px";
    hoverBox.style.left = "auto";
    hoverBox.style.right =
      ((fl.width || 0) - xa._offset - xa._length + 6) + "px";

    showVline(xv);
  });

  // Keep the box up while scrolling / panning (a redraw fires a spurious
  // unhover); otherwise hide after a short grace delay so crossing a pane
  // gap never makes it flicker.
  gd.on("plotly_unhover", function () {
    if (Date.now() < scrollUntil) { return; }
    clearTimeout(hideTimer);
    hideTimer = setTimeout(hideReadout, 140);
  });

})();
"""


def load_bets(csv_path: Path) -> pd.DataFrame:
    """Load team_taken.csv, keep only resolved bets, sorted oldest-first."""
    if not csv_path.exists():
        sys.exit(f"[erreur] Fichier introuvable : {csv_path}")
    df = pd.read_csv(csv_path)
    missing = {"timestamp", "result", "high", "low", "close"} - set(df.columns)
    if missing:
        sys.exit(f"[erreur] Colonnes manquantes dans le CSV : {sorted(missing)}")
    df = df[df["result"].isin(["win", "loss"])].copy()
    if df.empty:
        sys.exit("[erreur] Aucun pari resolu (win/loss) dans le fichier.")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def wilson_interval(wins: np.ndarray, n: np.ndarray, z: float = CONFIDENCE_Z):
    """Vectorized Wilson score interval (95 %). n must be >= 1 elementwise."""
    n = n.astype(float)
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return centre - margin, centre + margin


def rolling_wr_days(win: np.ndarray, ts_ms: np.ndarray, days: int) -> np.ndarray:
    """Rolling WR over a sliding time window of `days` days.

    For bet i, average the wins of every bet inside (ts[i] - days, ts[i]].
    Left NaN until the series has at least `days` of history. Mirrors
    rollingWR() in the post_script JS.
    """
    window_ms = int(days) * 86_400_000
    n = len(win)
    out = np.full(n, np.nan)
    lo = 0
    acc = 0
    for i in range(n):
        acc += int(win[i])
        while ts_ms[i] - ts_ms[lo] > window_ms:
            acc -= int(win[lo])
            lo += 1
        if ts_ms[i] - ts_ms[0] >= window_ms:
            out[i] = acc / (i - lo + 1)
    return out


def rolling_mean_days(values: np.ndarray, ts_ms: np.ndarray,
                      days: int) -> np.ndarray:
    """Rolling mean of `values` over a sliding time window of `days` days.

    For bet i, average every bet inside (ts[i] - days, ts[i]]. NaN values
    are skipped. The window grows from the first bet, so the line is full
    from the start (no left-NaN beyond what `values` itself carries)."""
    window_ms = int(days) * 86_400_000
    n = len(values)
    out = np.full(n, np.nan)
    lo = 0
    acc = 0.0
    cnt = 0
    for i in range(n):
        v = values[i]
        if not np.isnan(v):
            acc += v
            cnt += 1
        while ts_ms[i] - ts_ms[lo] > window_ms:
            lv = values[lo]
            if not np.isnan(lv):
                acc -= lv
                cnt -= 1
            lo += 1
        if cnt > 0:
            out[i] = acc / cnt
    return out


def rolling_count_days(ts_ms: np.ndarray, days: int) -> np.ndarray:
    """Count of bets in the trailing `days`-day window at each bet.

    For bet i, count every bet j with ts[i] - days < ts[j] <= ts[i] (bet i
    included). With days=7 this is the bets-per-week curve: how many bets the
    team placed in the week ending at each bet. No NaN -- the count is valid
    from the first bet (it just ramps up as the window fills)."""
    window_ms = int(days) * 86_400_000
    n = len(ts_ms)
    out = np.empty(n, dtype=float)
    lo = 0
    for i in range(n):
        while ts_ms[i] - ts_ms[lo] > window_ms:
            lo += 1
        out[i] = i - lo + 1
    return out


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> np.ndarray:
    """Volatility measure, min-max normalized to a 0-100 index.

    Base = the `period`-bar moving average of each bet bar's range
    (high - low). The bet bars are not time-contiguous, so the classic
    close-to-close True Range would spike at every gap; the intra-bar range
    is the clean measure. The series is then rescaled over the whole window
    to [0, 100] -- 0 = the calmest stretch, 100 = the most volatile."""
    rng = (df["high"] - df["low"]).astype(float)
    atr = rng.rolling(period, min_periods=period).mean().to_numpy()
    lo, hi = np.nanmin(atr), np.nanmax(atr)
    span = hi - lo
    if not (span > 0):
        return np.where(np.isnan(atr), np.nan, 50.0)
    return (atr - lo) / span * 100.0


def stake_fraction(capital: float) -> float:
    """Fraction of the at-risk balance to stake at this capital level.

    10 % under $1k, then 5 % / 2 % / 1 % at $1k / $10k / $50k, 3 % in the
    $100k-$1M band, and 1 % past $1M.
    """
    for threshold, frac in CAPITAL_TIERS:
        if capital >= threshold:
            return frac
    return CAPITAL_FRACTION_BASE


def compute_capital(win: np.ndarray, start: float = CAPITAL_START) -> np.ndarray:
    """Total capital after each bet : tiered stake + a profit-protection split.

    Two pots are tracked:
      - `risk`  : the at-risk balance -- the stake is a tier % of it.
      - `aside` : profit set aside, never staked again (but still wealth).
    `start` $ begins entirely in `risk`. Before each bet the stake is
    `stake_fraction(total) * risk`, with total = risk + aside.

      win  : profit = stake (the bet pays x2). Below CAPITAL_PROFIT_SPLIT_FROM
             all of it is reinvested into `risk`; at/above that total it is
             split 50/50 -- half into `risk`, half into `aside`.
      loss : the stake is forfeited from `risk`; `aside` is never touched.

    `capital[i]` = risk + aside after bet i -- the total wealth, which is
    what the chart's capital curve shows.
    """
    out = np.empty(len(win), dtype=float)
    risk, aside = float(start), 0.0
    for i in range(len(win)):
        total = risk + aside
        stake = stake_fraction(total) * risk
        if win[i] == 1:
            profit = stake
            if total >= CAPITAL_PROFIT_SPLIT_FROM:
                risk  += CAPITAL_PROFIT_REINVEST * profit
                aside += (1.0 - CAPITAL_PROFIT_REINVEST) * profit
            else:
                risk += profit
        else:
            risk -= stake
        out[i] = risk + aside
    return out


def load_warmup(test_start_ms: int, days: int = MAX_WARMUP_DAYS):
    """Run the team backtest over the `days` just before the test period.

    Those pre-test bets feed ONLY the rolling WR, so that curve is already at
    full window on bet 1 of the test period instead of starting `days` late.
    They sit in the model's validation span (between training and test), so
    they are a fair-enough warmup -- not bars the model was directly fit on.

    Returns (wins, timestamps_ms) as int arrays. Falls back to empty arrays if
    the model packages / parquet are unavailable (the rolling WR then simply
    starts `days` into the chart, as it did before).
    """
    try:
        _15min = Path(__file__).resolve().parent.parent.parent
        if str(_15min) not in sys.path:
            sys.path.insert(0, str(_15min))
        from modele_up import api as api_up
        from modele_down import api as api_down
        import detailed_log

        warm_start = test_start_ms - days * _DAY_MS
        warm_end = test_start_ms - 1
        up_full = api_up.evaluate_full_on_slice(warm_start, warm_end)
        down_full = api_down.evaluate_full_on_slice(warm_start, warm_end)
        per_bar = detailed_log.build_per_bar_table(up_full, down_full)
        taken = per_bar[(per_bar["bet_taken"] == 1)
                        & per_bar["result"].isin(["win", "loss"])]
        if taken.empty:
            print("  [warmup] aucun pari avant la periode de test.")
            return np.empty(0, dtype=int), np.empty(0, dtype="int64")
        taken = taken.sort_values("timestamp")
        w_win = (taken["result"] == "win").to_numpy().astype(int)
        w_ts = taken["timestamp"].to_numpy().astype("int64")
        print(f"  [warmup] {len(w_win)} paris charges "
              f"({days} j avant le test).")
        return w_win, w_ts
    except Exception as exc:
        print(f"  [warmup] indisponible ({type(exc).__name__}: {exc}).")
        return np.empty(0, dtype=int), np.empty(0, dtype="int64")


def compute_series(df: pd.DataFrame, warmup: tuple) -> dict:
    """Build every curve from the win/loss sequence and the bet OHLC.

    `warmup` = (wins, timestamps_ms) for the bets just before the test period.
    It feeds ONLY the rolling WR (so that curve is full from bet 1); every
    other curve stays test-period only.
    """
    win = (df["result"] == "win").to_numpy().astype(int)
    ts_ms = df["timestamp"].to_numpy().astype("int64")
    n = win.size
    bet_no = np.arange(1, n + 1)
    cum_wins = np.cumsum(win)
    cum_losses = bet_no - cum_wins
    cum_wr = cum_wins / bet_no

    # Headline curve : cumulative WR with its thin-sample head shrunk toward
    # 50 % -- (cum + a) / (bets + 2a). The prior is worth ~2a pseudo-bets, so
    # the first weeks no longer swing to 0/100 %, and it washes out to a
    # sub-0.5 % nudge within a month; the rest is the true cumulative WR.
    head_wr = (cum_wins + HEAD_PRIOR) / (bet_no + 2 * HEAD_PRIOR)

    # Rolling WR : computed over warmup + test bets so it is already at full
    # window on bet 1 of the test period; only the test slice is kept.
    w_win, w_ts = warmup
    roll_win = np.concatenate([w_win, win])
    roll_ts = np.concatenate([w_ts, ts_ms])
    test_offset = int(w_win.size)
    rolling = rolling_wr_days(roll_win, roll_ts,
                              DEFAULT_ROLLING_DAYS)[test_offset:]

    ci_lo, ci_hi = wilson_interval(cum_wins, bet_no)
    price = df["close"].to_numpy(dtype=float)
    atr = compute_atr(df, ATR_PERIOD)
    atr_ma = rolling_mean_days(atr, ts_ms, ATR_MA_DAYS)
    capital = compute_capital(win)
    bets_week = rolling_count_days(ts_ms, 7)   # bets in the trailing 7 days

    return dict(win=win, ts_ms=ts_ms, n=n, bet_no=bet_no, cum_wins=cum_wins,
                cum_losses=cum_losses, cum_wr=cum_wr, head_wr=head_wr,
                rolling=rolling, ci_lo=ci_lo, ci_hi=ci_hi,
                price=price, atr=atr, atr_ma=atr_ma, capital=capital,
                bets_week=bets_week,
                roll_win=roll_win, roll_ts=roll_ts, test_offset=test_offset)


def _padded_range(values: np.ndarray, frac: float = 0.12) -> list[float]:
    """[min, max] of `values` (NaN-safe) widened by `frac` on each side."""
    v = values[~np.isnan(values)] if values.dtype.kind == "f" else values
    if v.size == 0:
        return [0.0, 1.0]
    lo, hi = float(v.min()), float(v.max())
    pad = (hi - lo) * frac or abs(hi) * 0.02 or 1.0
    return [lo - pad, hi + pad]


def _padded_log_range(values: np.ndarray, frac: float = 0.04) -> list[float]:
    """log10 [min, max] of the positive `values`, widened by `frac` each side.
    Plotly log axes take their `range` in log10 units."""
    v = values[np.isfinite(values) & (values > 0)]
    if v.size == 0:
        return [0.0, 1.0]
    lo, hi = np.log10(float(v.min())), np.log10(float(v.max()))
    pad = (hi - lo) * frac or 0.3
    return [lo - pad, hi + pad]


def build_figure(df: pd.DataFrame, s: dict):
    """Assemble the Plotly figure. Plotly is imported lazily."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        sys.exit("[erreur] plotly n'est pas installe.  ->  pip install plotly")

    x = df["dt"].to_numpy()
    final_wr = float(s["cum_wr"][-1])

    fig = go.Figure()

    # --- trace 0 : Wilson 95 % band (closed polygon, toggled off) ----------
    band_x = np.concatenate([x, x[::-1]])
    band_y = np.concatenate([s["ci_hi"], s["ci_lo"][::-1]])
    fig.add_trace(go.Scatter(
        x=band_x, y=band_y, fill="toself", fillcolor=C["band"],
        line=dict(width=0), mode="lines", name="Intervalle 95 % (Wilson)",
        visible="legendonly", hoverinfo="skip",
    ))

    # --- trace 1 : 50 % baseline (also the fill floor for the WR line) -----
    fig.add_trace(go.Scatter(
        x=x, y=np.full(s["n"], 0.5), mode="lines",
        line=dict(color=C["baseline"], width=1, dash="dash"),
        name="Baseline 50 %", showlegend=False, hoverinfo="skip",
    ))

    # --- trace 2 : cumulative WR (headline) -- thin-sample head shrunk -----
    fig.add_trace(go.Scatter(
        x=x, y=s["head_wr"], mode="lines",
        line=dict(color=C["smoothed"], width=2.6),
        fill="tonexty", fillcolor=C["edge_fill"],
        name="WR cumulatif", hovertemplate="%{y}<extra></extra>",
    ))

    # --- trace 3 : rolling WR (window editable in the panel) --------------
    fig.add_trace(go.Scatter(
        x=x, y=s["rolling"], mode="lines", connectgaps=False,
        line=dict(color=C["rolling"], width=1.0),
        name="WR glissant", hovertemplate="%{y}<extra></extra>",
    ))

    # --- trace 4 : BTC price. Parked on the main axes; render() moves it
    #               to its own pane (x2/y2) when the user opens it. --------
    fig.add_trace(go.Scatter(
        x=x, y=s["price"], mode="lines", xaxis="x", yaxis="y",
        line=dict(color=C["price"], width=1.4),
        name="Prix BTC", visible="legendonly",
        hovertemplate="%{y}<extra></extra>",
    ))

    # --- trace 5 : ATR. Parked on the main axes; render() moves it to
    #               its own pane (x3/y3) when the user opens it. ----------
    fig.add_trace(go.Scatter(
        x=x, y=s["atr_ma"], mode="lines", xaxis="x", yaxis="y", connectgaps=False,
        line=dict(color=C["atr"], width=1.6),
        name="ATR glissant", visible="legendonly",
        hovertemplate="%{y}<extra></extra>",
    ))

    # --- trace 6 : total capital. $500 start, tiered stake (10/5/2/1 % up
    #               to $50k, 3 % in $100k-$1M, 1 % above); from $100k each
    #               win is split 50/50 (half reinvested, half set aside).
    #               Parked on main axes; render() -> pane x4/y4 (LOG). ----
    fig.add_trace(go.Scatter(
        x=x, y=s["capital"], mode="lines", xaxis="x", yaxis="y",
        line=dict(color=C["capital"], width=1.6),
        name="Capital (500 $, paliers)", visible="legendonly",
        hovertemplate="%{y}<extra></extra>",
    ))

    # --- trace 7 : bets per week. Trailing 7-day count at each bet.
    #               Parked on the main axes; render() moves it to its own
    #               pane (x5/y5) when the user opens it. -------------------
    fig.add_trace(go.Scatter(
        x=x, y=s["bets_week"], mode="lines", xaxis="x", yaxis="y",
        line=dict(color=C["bets"], width=1.4, shape="hv"),
        name="Paris / semaine", visible="legendonly",
        hovertemplate="%{y}<extra></extra>",
    ))

    # --- final WR reference line (annotation index 0) ---------------------
    fig.add_hline(
        y=final_wr, line=dict(color=C["final"], width=1, dash="dot"),
        annotation_text=f" WR final {final_wr:.2%} ",
        annotation_position="top left",
        annotation_font=dict(color=C["final"], size=11),
    )
    # --- pane name tags (annotation index 1 = price, 2 = atr) -------------
    fig.add_annotation(
        text="Prix BTC", xref="x2 domain", yref="y2 domain",
        x=0.008, y=0.94, xanchor="left", yanchor="top", showarrow=False,
        font=dict(color=C["price"], size=10.5), visible=False,
    )
    fig.add_annotation(
        text="ATR glissant — moyenne normalisée (0-100)",
        xref="x3 domain", yref="y3 domain",
        x=0.008, y=0.94, xanchor="left", yanchor="top", showarrow=False,
        font=dict(color=C["atr"], size=10.5), visible=False,
    )
    # --- pane name tag (annotation index 3 = capital) --------------------
    fig.add_annotation(
        text="Capital — 500 $, paliers 10/5/2/1 % puis 3 %/1 %, >=100k : 50/50 (log)",
        xref="x4 domain", yref="y4 domain",
        x=0.008, y=0.94, xanchor="left", yanchor="top", showarrow=False,
        font=dict(color=C["capital"], size=10.5), visible=False,
    )
    # --- pane name tag (annotation index 4 = bets per week) --------------
    fig.add_annotation(
        text="Paris / semaine — comptage glissant sur 7 jours",
        xref="x5 domain", yref="y5 domain",
        x=0.008, y=0.94, xanchor="left", yanchor="top", showarrow=False,
        font=dict(color=C["bets"], size=10.5), visible=False,
    )

    # --- initial axis windows --------------------------------------------
    # WR axis is locked to 50-70 % : a stable, readable scale rather than one
    # that auto-fits the data extent. The price / ATR panes still auto-fit.
    y_range = [0.50, 0.70]
    y2_range = _padded_range(s["price"])
    y3_range = [-4.0, 104.0]                     # ATR is a fixed 0-100 index
    y4_range = _padded_log_range(s["capital"])   # log10 units (log axis)
    y5_range = _padded_range(s["bets_week"])     # bets / week (linear count)

    d0 = df["dt"].iloc[0].strftime("%d/%m/%Y")
    d1 = df["dt"].iloc[-1].strftime("%d/%m/%Y")

    grid = dict(gridcolor=C["grid"], griddash="dot", zeroline=False,
                linecolor=C["axis"], tickcolor=C["axis"])

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        font=dict(family='"Segoe UI", Roboto, "Helvetica Neue", sans-serif',
                  color=C["text"], size=12),
        title=dict(
            text=("<b>Évolution du Win Rate — Équipe 15 min</b>"
                  f"<br><span style='font-size:13px;color:{C['dim']};"
                  f"font-weight:400'>{s['n']} paris   ·   {d0} → {d1}   ·   "
                  f"WR final {final_wr:.2%}</span>"),
            x=0.012, xanchor="left", y=0.975, yanchor="top",
            font=dict(size=19, color=C["text"]),
        ),
        hovermode="x unified",
        dragmode="pan",
        showlegend=False,                       # replaced by the control panel
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=C["dim"],
                     activecolor=C["smoothed"]),
        newshape=dict(line=dict(color=C["smoothed"], width=1.5)),
        margin=dict(l=64, r=30, t=104, b=40),

        # --- main X axis : carries the range buttons --------------------
        xaxis=dict(
            domain=[0.0, 1.0], anchor="y", showspikes=False,
            hoverformat="%a %d %b %Y · %H:%M", rangeslider=dict(visible=False),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1A", step="year", stepmode="backward"),
                    dict(step="all", label="Tout"),
                ],
                bgcolor="#1c2030", activecolor="#2962ff",
                font=dict(color=C["text"], size=11),
                bordercolor=C["axis"], borderwidth=1,
                x=0.012, xanchor="left", y=1.022, yanchor="bottom",
            ),
            **grid,
        ),
        # --- pane X axes : locked to the main X via matches -------------
        xaxis2=dict(domain=[0.0, 1.0], anchor="y2", matches="x",
                    visible=False, showspikes=False,
                    hoverformat="%a %d %b %Y · %H:%M", **grid),
        xaxis3=dict(domain=[0.0, 1.0], anchor="y3", matches="x",
                    visible=False, showspikes=False,
                    hoverformat="%a %d %b %Y · %H:%M", **grid),
        xaxis4=dict(domain=[0.0, 1.0], anchor="y4", matches="x",
                    visible=False, showspikes=False,
                    hoverformat="%a %d %b %Y · %H:%M", **grid),
        xaxis5=dict(domain=[0.0, 1.0], anchor="y5", matches="x",
                    visible=False, showspikes=False,
                    hoverformat="%a %d %b %Y · %H:%M", **grid),

        # --- main Y axis : win rate. fixedrange -> JS-driven -----------
        yaxis=dict(
            domain=[0.0, 1.0], anchor="x",
            tickformat=".0%", hoverformat=".2%", range=y_range,
            fixedrange=True, dtick=0.01, showspikes=False, **grid,
        ),
        # --- pane Y axis : BTC price (domain set by render() on open) ---
        yaxis2=dict(
            domain=[0.0, 1.0], anchor="x2", fixedrange=True, visible=False,
            range=y2_range, tickprefix="$", tickformat=",.0f",
            showspikes=False, **grid,
        ),
        # --- pane Y axis : ATR (domain set by render() on open) ---------
        yaxis3=dict(
            domain=[0.0, 1.0], anchor="x3", fixedrange=True, visible=False,
            range=y3_range, tickformat=".0f", dtick=20,
            showspikes=False, **grid,
        ),
        # --- pane Y axis : compound capital. LOG scale -- the balance
        #     compounds over thousands of bets, so a linear axis would be a
        #     flat-then-vertical hockey stick. range is in log10 units;
        #     render() auto-fits it (log-aware) on open and on zoom.
        yaxis4=dict(
            domain=[0.0, 1.0], anchor="x4", type="log", fixedrange=True,
            visible=False, range=y4_range, tickprefix="$",
            exponentformat="power", showspikes=False, **grid,
        ),
        # --- pane Y axis : bets / week (linear count, auto-fit on open) --
        yaxis5=dict(
            domain=[0.0, 1.0], anchor="x5", fixedrange=True, visible=False,
            range=y5_range, tickformat=".0f",
            showspikes=False, **grid,
        ),
    )
    return fig


def write_chart(fig, out_path: Path, s: dict) -> None:
    """Write the self-contained interactive HTML.

    The win/loss + timestamp arrays are embedded as a JS prelude so the
    rolling-WR window can be recomputed in the browser on demand.
    """
    def _arr(a):
        # Embed as a JS array literal. NaN survives as the JS `NaN` global,
        # which is exactly what we want (it marks "no value" for that bet).
        return json.dumps([float(v) for v in a])

    # Every per-trace Y series is embedded as a plain JS array. Plotly stores
    # numpy-backed arrays in a base64 typed-array form, so gd.data[i].y[idx]
    # is unreadable from JS -- these arrays are the source of truth instead.
    data_js = (
        # ROLL_* span warmup + test bets. rollingWR() slides over them and
        # keeps only the test slice (TEST_OFFSET onward), so the rolling WR is
        # already at full window on the first test bet. WR_TS = the test
        # timestamps, used to map a hovered X to a bet index.
        "var ROLL_WINS=" + json.dumps([int(v) for v in s["roll_win"]]) + ";\n"
        "var ROLL_TS=" + json.dumps([int(v) for v in s["roll_ts"]]) + ";\n"
        "var TEST_OFFSET=" + str(s["test_offset"]) + ";\n"
        "var WR_TS=ROLL_TS.slice(TEST_OFFSET);\n"
        "var DEFAULT_DAYS=" + str(DEFAULT_ROLLING_DAYS) + ";\n"
        "var WR_CUM=" + _arr(s["head_wr"]) + ";\n"
        "var WR_PRICE=" + _arr(s["price"]) + ";\n"
        "var WR_ATR=" + _arr(s["atr"]) + ";\n"
        "var ATR_DAYS=" + str(ATR_MA_DAYS) + ";\n"
        "var WR_CAPITAL=" + _arr(s["capital"]) + ";\n"
        "var WR_BETS=" + _arr(s["bets_week"]) + ";\n"
    )
    config = dict(
        # scrollZoom OFF: the wheel is handled by our own listener, which
        # zooms X and every active Y axis in one relayout.
        scrollZoom=False,
        displaylogo=False,
        displayModeBar=True,
        responsive=True,
        doubleClick="reset",
        modeBarButtonsToAdd=["drawline", "drawopenpath", "eraseshape"],
        modeBarButtonsToRemove=["lasso2d", "select2d"],
        toImageButtonOptions=dict(format="png", filename="wr_evolution",
                                  height=900, width=1600, scale=2),
    )
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True,
                   config=config, post_script=data_js + _AUTO_Y_JS)


def print_summary(df: pd.DataFrame, s: dict, out_path: Path) -> None:
    """Console recap of the run."""
    wins = int(s["cum_wins"][-1])
    n = s["n"]
    print("\n  Graphique généré :", out_path)
    print(f"  {'Paris analysés':<20}: {n}  ({wins} W / {n - wins} L)")
    print(f"  {'Période':<20}: {df['dt'].iloc[0]:%d/%m/%Y} → "
          f"{df['dt'].iloc[-1]:%d/%m/%Y}")
    if s["test_offset"] > 0:
        print(f"  {'Warmup pré-test':<20}: {s['test_offset']} paris  "
              f"(moyenne glissante pleine dès le 1er pari)")
    print(f"  {'WR final':<20}: {s['cum_wr'][-1]:.2%}")
    roll = s["rolling"]
    if not np.isnan(roll).all():
        bi, wi = int(np.nanargmax(roll)), int(np.nanargmin(roll))
        print(f"  {'Meilleure forme':<20}: {roll[bi]:.2%}  "
              f"(glissant {DEFAULT_ROLLING_DAYS} j, {df['dt'].iloc[bi]:%d/%m/%Y})")
        print(f"  {'Pire forme':<20}: {roll[wi]:.2%}  "
              f"(glissant {DEFAULT_ROLLING_DAYS} j, {df['dt'].iloc[wi]:%d/%m/%Y})")
    print(f"  {'Prix BTC (fin)':<20}: ${s['price'][-1]:,.0f}")
    atr = s["atr_ma"][~np.isnan(s["atr_ma"])]
    if atr.size:
        print(f"  {'ATR glissant (moy.)':<20}: {atr.mean():.1f}")
    bw = s["bets_week"]
    print(f"  {'Paris / semaine':<20}: {bw.mean():.1f} moy.  "
          f"(min {bw.min():.0f}, max {bw.max():.0f})")
    cap_final = float(s["capital"][-1])
    mult = cap_final / CAPITAL_START
    cap_str = f"{cap_final:,.2f}" if cap_final < 1e9 else f"{cap_final:.3e}"
    mult_str = f"{mult:,.1f}" if mult < 1e9 else f"{mult:.3e}"
    print(f"  {'Capital final':<20}: ${cap_str}  "
          f"(x{mult_str} depuis ${CAPITAL_START:.0f}, paliers + 50/50 des 100k)")


def main() -> None:
    csv_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_CSV
    print(f"Lecture : {csv_path}")
    df = load_bets(csv_path)
    print(f"Warmup  : prediction des {MAX_WARMUP_DAYS} j avant le test "
          f"(peut prendre ~15 s)...")
    warmup = load_warmup(int(df["timestamp"].iloc[0]), MAX_WARMUP_DAYS)
    series = compute_series(df, warmup)
    fig = build_figure(df, series)
    write_chart(fig, OUTPUT_HTML, series)
    print_summary(df, series, OUTPUT_HTML)
    try:
        webbrowser.open(OUTPUT_HTML.as_uri())
        print("  Ouverture dans le navigateur...")
    except Exception:
        print("  (ouverture auto impossible — ouvrir le fichier manuellement)")


if __name__ == "__main__":
    main()
