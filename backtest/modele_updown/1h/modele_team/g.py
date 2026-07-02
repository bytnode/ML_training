"""g.py — graphique interactif de l'evolution du DERNIER backtest trading.

Lit logs/trades.csv (un row par trade simule, ecrase a chaque `python main.py`)
et rend un dashboard HTML sombre et autonome (Plotly, ouvert automatiquement
dans le navigateur) :
  - capital compose ($), avec les trades gagnants/perdants marques
  - drawdown (%)
  - win rate cumulee nette (tous / UP / DOWN)
  - P/L mensuel (%)

Les calculs (capital compose, drawdown, EV/trade) reprennent exactement la
formule de trading_stats() dans main.py (meme INITIAL_CAPITAL / EXPOSURE),
donc les chiffres du header correspondent au rapport Rich du dernier run.

Working-directory contract (comme main.py) :
    cd backtest/modele_updown/1h/modele_team
    python main.py     # produit logs/trades.csv
    python g.py         # affiche son evolution
"""
from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
TRADES_CSV = ROOT / "logs" / "trades.csv"
OUTPUT_HTML = ROOT / "logs" / "g_evolution.html"

from config_main import INITIAL_CAPITAL, EXPOSURE

C = dict(
    bg="#131722", grid="#222631", text="#d1d4dc", dim="#787b86",
    equity="#3b9aff", equity_fill="rgba(59,154,255,0.10)",
    dd="#ef5350", dd_fill="rgba(239,83,80,0.15)",
    wr="#26a69a", wr_up="#4dd0e1", wr_down="#ffb74d",
    win="#26a69a", loss="#ef5350",
)


def load_trades() -> pd.DataFrame:
    if not TRADES_CSV.exists():
        raise FileNotFoundError(f"{TRADES_CSV} introuvable. Lance d'abord `python main.py`.")
    df = pd.read_csv(TRADES_CSV)
    if df.empty:
        raise ValueError(f"{TRADES_CSV} est vide (aucun trade sur la derniere fenetre).")
    df = df.sort_values("traded_ts").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["traded_ts"], unit="ms", utc=True)
    return df


def compute_curves(df: pd.DataFrame) -> dict:
    net = df["net_ret"].to_numpy(dtype=float) * EXPOSURE
    equity = INITIAL_CAPITAL * np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100.0

    win = df["win_net"].to_numpy(dtype=bool)
    cum_wr = np.cumsum(win) / (np.arange(len(win)) + 1) * 100.0

    up_mask = (df["direction"] == "up").to_numpy()
    cum_wr_up = np.full(len(win), np.nan)
    cum_wr_down = np.full(len(win), np.nan)
    if up_mask.any():
        idx = np.flatnonzero(up_mask)
        cum_wr_up[idx] = np.cumsum(win[idx]) / (np.arange(len(idx)) + 1) * 100.0
    if (~up_mask).any():
        idx = np.flatnonzero(~up_mask)
        cum_wr_down[idx] = np.cumsum(win[idx]) / (np.arange(len(idx)) + 1) * 100.0

    months = df["dt"].dt.strftime("%Y-%m")
    equity_prev = np.concatenate(([INITIAL_CAPITAL], equity[:-1]))
    m_start, m_end = {}, {}
    for m, e0, e1 in zip(months, equity_prev, equity):
        if m not in m_start:
            m_start[m] = e0
        m_end[m] = e1
    month_labels = sorted(m_start)
    month_pnl = [m_end[m] - m_start[m] for m in month_labels]
    month_pct = [(m_end[m] / m_start[m] - 1.0) * 100.0 for m in month_labels]

    return dict(
        equity=equity, drawdown=drawdown, cum_wr=cum_wr,
        cum_wr_up=cum_wr_up, cum_wr_down=cum_wr_down,
        month_labels=month_labels, month_pnl=month_pnl, month_pct=month_pct,
        net=net,
    )


def headline_stats(df: pd.DataFrame, curves: dict) -> str:
    net = curves["net"]
    equity = curves["equity"]
    final = float(equity[-1])
    n = len(net)
    wins, losses = net[net > 0], net[net < 0]
    wr = len(wins) / n if n else 0.0
    lr = len(losses) / n if n else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(-losses.mean()) if len(losses) else 0.0
    ev = (wr * avg_win - lr * avg_loss) * 100.0
    net_total_pct = (final / INITIAL_CAPITAL - 1.0) * 100.0
    max_dd = float(curves["drawdown"].min())
    start, end = df["dt"].iloc[0].strftime("%Y-%m-%d"), df["dt"].iloc[-1].strftime("%Y-%m-%d")
    return (
        f"<b>P/L NET {net_total_pct:+.1f}%</b>  ({final - INITIAL_CAPITAL:+,.0f}$ sur {INITIAL_CAPITAL:,.0f}$)"
        f"   |   {n} trades, WR {wr*100:.1f}%, EV/trade {ev:+.3f}%"
        f"   |   max DD {max_dd:.1f}%"
        f"   |   {start} -> {end}"
    )


def build_figure(df: pd.DataFrame, curves: dict, title_text: str) -> go.Figure:
    x = df["dt"]
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.42, 0.16, 0.16, 0.26], vertical_spacing=0.04,
        subplot_titles=("Capital compose ($)", "Drawdown (%)",
                         "Win rate cumulee nette (%)", "P/L mensuel (%)"),
    )

    fig.add_trace(go.Scatter(
        x=x, y=curves["equity"], mode="lines", name="Capital",
        line=dict(color=C["equity"], width=1.6),
        fill="tozeroy", fillcolor=C["equity_fill"],
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>capital=%{y:,.0f}$<extra></extra>",
    ), row=1, col=1)

    wins = df["win_net"]
    fig.add_trace(go.Scatter(
        x=x[wins], y=curves["equity"][wins.to_numpy()], mode="markers",
        name="trade gagnant", marker=dict(color=C["win"], size=4, opacity=0.6),
        hovertemplate="%{x|%Y-%m-%d %H:%M} WIN<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x[~wins], y=curves["equity"][(~wins).to_numpy()], mode="markers",
        name="trade perdant", marker=dict(color=C["loss"], size=4, opacity=0.6),
        hovertemplate="%{x|%Y-%m-%d %H:%M} LOSS<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=curves["drawdown"], mode="lines", name="Drawdown",
        line=dict(color=C["dd"], width=1.2),
        fill="tozeroy", fillcolor=C["dd_fill"],
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>DD=%{y:.2f}%<extra></extra>",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=curves["cum_wr"], mode="lines", name="WR cumulee (tous)",
        line=dict(color=C["wr"], width=1.6),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>WR=%{y:.2f}%<extra></extra>",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=curves["cum_wr_up"], mode="lines", name="WR cumulee UP",
        line=dict(color=C["wr_up"], width=1, dash="dot"), connectgaps=True,
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>WR UP=%{y:.2f}%<extra></extra>",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=curves["cum_wr_down"], mode="lines", name="WR cumulee DOWN",
        line=dict(color=C["wr_down"], width=1, dash="dot"), connectgaps=True,
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>WR DOWN=%{y:.2f}%<extra></extra>",
    ), row=3, col=1)
    fig.add_hline(y=50, line=dict(color=C["dim"], width=1, dash="dash"), row=3, col=1)

    month_x = pd.to_datetime(curves["month_labels"])
    month_colors = [C["win"] if v >= 0 else C["loss"] for v in curves["month_pnl"]]
    fig.add_trace(go.Bar(
        x=month_x, y=curves["month_pct"], name="P/L mensuel (%)",
        marker_color=month_colors,
        hovertemplate="%{x|%Y-%m}<br>%{y:+.1f}%<extra></extra>",
    ), row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        font=dict(color=C["text"], size=12),
        title=dict(text=title_text, x=0.01, xanchor="left", font=dict(size=14)),
        height=1000, hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=0.01),
        margin=dict(t=110, l=60, r=30, b=40),
        dragmode="pan",   # glisser = pan ; le zoom se fait a la molette (scrollZoom, cf write_html)
    )
    fig.update_xaxes(gridcolor=C["grid"])
    fig.update_yaxes(gridcolor=C["grid"])
    return fig


# Poignees de redimensionnement entre panneaux : glisser la ligne separant
# 2 panneaux vers le haut/bas agrandit l'un et retrecit l'autre (les 2 autres
# panneaux ne bougent pas). GAP doit rester identique a vertical_spacing
# (build_figure). {plot_id} est substitue automatiquement par write_html().
RESIZE_PANELS_JS = """
(function () {
  var gd = document.getElementById('{plot_id}');
  if (!gd || !gd.on) { return; }

  var ROWS = 4;
  var GAP = 0.04;        // == vertical_spacing dans build_figure
  var MIN_FRAC = 0.05;   // hauteur mini d'un panneau (fraction de la zone de plot)
  var Y_AXES = ['yaxis', 'yaxis2', 'yaxis3', 'yaxis4'];
  var TITLES = ['Capital compose ($)', 'Drawdown (%)',
                'Win rate cumulee nette (%)', 'P/L mensuel (%)'];

  function getDomains() {
    var fl = gd._fullLayout;
    return Y_AXES.map(function (k) { return fl[k].domain.slice(); });
  }

  function applyDomains(domains) {
    var upd = {};
    for (var i = 0; i < ROWS; i++) { upd[Y_AXES[i] + '.domain'] = domains[i]; }
    var anns = (gd.layout.annotations || []).map(function (a) {
      var idx = TITLES.indexOf(a.text);
      if (idx >= 0) { a = Object.assign({}, a, { y: domains[idx][1] }); }
      return a;
    });
    upd.annotations = anns;
    Plotly.relayout(gd, upd);
  }

  function positionHandles() {
    var sz = gd._fullLayout._size;
    var domains = getDomains();
    document.querySelectorAll('.g-resize-handle').forEach(function (h) {
      var i = h._rowAbove;
      var mid = (domains[i][0] + domains[i + 1][1]) / 2;
      var yPx = sz.t + (1 - mid) * sz.h;
      h.style.top = (yPx - 5) + 'px';
    });
  }

  function buildHandles() {
    gd.style.position = 'relative';
    document.querySelectorAll('.g-resize-handle').forEach(function (h) { h.remove(); });
    for (var i = 0; i < ROWS - 1; i++) {
      var handle = document.createElement('div');
      handle.className = 'g-resize-handle';
      handle.title = 'Glisser pour redimensionner ce panneau';
      handle._rowAbove = i;
      Object.assign(handle.style, {
        position: 'absolute', left: '50px', right: '10px', height: '10px',
        cursor: 'ns-resize', zIndex: 1000, background: 'transparent',
        borderTop: '1px solid rgba(120,123,134,0.25)',
      });
      handle.addEventListener('mouseenter', function (ev) {
        ev.target.style.background = 'rgba(59,154,255,0.35)';
      });
      handle.addEventListener('mouseleave', function (ev) {
        if (!dragging) { ev.target.style.background = 'transparent'; }
      });
      gd.appendChild(handle);
    }
    positionHandles();
  }

  var dragging = null;
  document.addEventListener('mousedown', function (ev) {
    if (!ev.target.classList || !ev.target.classList.contains('g-resize-handle')) { return; }
    var i = ev.target._rowAbove;
    var domains = getDomains();
    dragging = {
      i: i, handle: ev.target, startY: ev.clientY,
      startMid: (domains[i][0] + domains[i + 1][1]) / 2,
      domains: domains,
    };
    ev.preventDefault();
  });
  document.addEventListener('mousemove', function (ev) {
    if (!dragging) { return; }
    var sz = gd._fullLayout._size;
    var dFrac = -(ev.clientY - dragging.startY) / sz.h;
    var i = dragging.i;
    var domains = dragging.domains.map(function (d) { return d.slice(); });
    var midMin = domains[i + 1][0] + MIN_FRAC + GAP / 2;
    var midMax = domains[i][1] - MIN_FRAC - GAP / 2;
    var mid = Math.max(midMin, Math.min(midMax, dragging.startMid + dFrac));
    domains[i][0] = mid + GAP / 2;
    domains[i + 1][1] = mid - GAP / 2;
    applyDomains(domains);
    positionHandles();
  });
  document.addEventListener('mouseup', function () {
    if (dragging) { dragging.handle.style.background = 'transparent'; }
    dragging = null;
  });

  gd.on('plotly_relayout', positionHandles);
  window.addEventListener('resize', positionHandles);
  setTimeout(buildHandles, 0);
})();
"""


def main() -> None:
    df = load_trades()
    curves = compute_curves(df)
    title_text = headline_stats(df, curves)
    fig = build_figure(df, curves, title_text)
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUTPUT_HTML, include_plotlyjs=True,
                    config={"scrollZoom": True, "responsive": True},
                    post_script=RESIZE_PANELS_JS)
    print(f"Graphique ecrit : {OUTPUT_HTML}  ({len(df)} trades)")
    webbrowser.open(OUTPUT_HTML.as_uri())


if __name__ == "__main__":
    main()
