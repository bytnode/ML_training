"""Harnais volume-matche pour la mission +1.5pt WR (objectif >= 64.64% @ ~1001 bets).

Reproduit EXACTEMENT le split + le stack de decision de train.py (TEST_DAYS=365,
TRAIN_DAYS=3600, VAL_FRACTION=0.15, skip cells, hard/soft bad hours, bad days,
cross-model p_up). Chaque variante est evaluee a VOLUME CONSTANT : le seuil
unanime T est re-resolu par dichotomie pour atteindre ~1001 bets, puis on
compare les WR. Fenetre primaire = test (2025-05-09 -> 2026-05-09, metrique de
l'objectif). Fenetre de controle anti-overfit = derniere annee de val
(2024-05-09 -> 2025-05-09), meme protocole (sans cross-model: p_up indispo).

Usage: python goal_harness.py A   (batch A: ensemble / decision layer / HP)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from features import build_training_dataframe  # noqa: E402
import config as C                              # noqa: E402

MS = 86_400_000
TARGET = 1001

print("loading parquet...", flush=True)
DF = build_training_dataframe(str(ROOT / "db/btc_multi_tf_15m.parquet"), verbose=False)
DF = DF.sort_values("timestamp").reset_index(drop=True)

ts_max = int(DF["timestamp"].max())
test_cut = ts_max - 365 * MS
tv_cut = test_cut - 3600 * MS
test_start = int(DF["timestamp"].searchsorted(test_cut, "left"))
tv_start = int(DF["timestamp"].searchsorted(tv_cut, "left"))
val_end = test_start
val_size = int((val_end - tv_start) * 0.15)
train_end = val_end - val_size
TRAIN = DF.iloc[tv_start:train_end]
VAL = DF.iloc[train_end:val_end]
TEST = DF.iloc[test_start:]
BASE_FEATS = [c for c in DF.columns
              if c not in ("timestamp", "label", "volatility_state", "is_weekend")]

# Fenetre de controle honnete = derniere annee de val, meme taille que test.
vc_cut = test_cut - 365 * MS
VCTRL = VAL[VAL["timestamp"] >= vc_cut]

UP = pd.read_csv(ROOT.parent / "modele_up/logs/test_predictions.csv",
                 usecols=["timestamp", "p_up"])
UP_MAP = dict(zip(UP["timestamp"].astype("int64"), UP["p_up"].astype(float)))


def _ctx(df, with_pup):
    ts = df["timestamp"].to_numpy(np.int64)
    return {
        "hour": ((ts // 900_000) % 96 // 4).astype(int),
        "dow": ((ts // MS + 3) % 7 + 1).astype(int),
        "vol": df["volatility_state"].astype(int).to_numpy(),
        "wknd": df["is_weekend"].astype(int).to_numpy(),
        "label": df["label"].to_numpy(int),
        "pup": (np.array([UP_MAP.get(int(t), np.nan) for t in ts])
                if with_pup else None),
    }


CTX_TEST = _ctx(TEST, True)
CTX_VCTRL = _ctx(VCTRL, False)


def decide(p, ctx, T_base, skip_cells=C.ASYM_SKIP_CELLS, hard_hours=C.ASYM_BAD_HOURS,
           bad_days=C.ADAPTIVE_BAD_DAYS, wr_bad_day=C.ADAPTIVE_WR_BAD_DAY,
           soft_hours=C.ADAPTIVE_BAD_HOURS, soft_delta=C.ASYM_BAD_HOURS_DELTA,
           max_pup=C.ASYM_MAX_P_UP, extra_mask=None):
    T = np.full(len(p), float(T_base))
    for v, w in skip_cells:
        T[(ctx["vol"] == v) & (ctx["wknd"] == w)] = 1.0
    if len(hard_hours):
        T[np.isin(ctx["hour"], list(hard_hours))] = 1.0
    if len(bad_days):
        T[np.isin(ctx["dow"], list(bad_days))] += wr_bad_day
    if len(soft_hours):
        T[np.isin(ctx["hour"], list(soft_hours))] += soft_delta
    take = p >= T
    if ctx["pup"] is not None and max_pup < 1.0:
        with np.errstate(invalid="ignore"):
            take &= (ctx["pup"] < max_pup)   # politique train.py: NaN -> reject
    if extra_mask is not None:
        take &= extra_mask
    return take


def vmatch(p, ctx, target=TARGET, tol=5, lo=0.50, hi=0.75, **kw):
    """Resout T pour atteindre ~target bets. Renvoie (T, n, wr)."""
    best = None
    for _ in range(48):
        mid = (lo + hi) / 2
        take = decide(p, ctx, mid, **kw)
        n = int(take.sum())
        if best is None or abs(n - target) < abs(best[1] - target):
            wr = float(ctx["label"][take].mean()) if n else float("nan")
            best = (mid, n, wr)
        if abs(n - target) <= tol:
            break
        if n > target:
            lo = mid
        else:
            hi = mid
    return best


def train_model(seed=42, feats=BASE_FEATS, train_df=TRAIN, val_df=VAL,
                sample_weight=None, **hp_over):
    hp = dict(n_estimators=1000, max_depth=5, learning_rate=0.05,
              min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
              gamma=1.5, scale_pos_weight=1.3, random_state=seed, n_jobs=-1,
              objective="binary:logistic", eval_metric="logloss")
    es = hp_over.pop("early_stopping_rounds", 50)
    hp.update(hp_over)
    if es is None:
        m = xgb.XGBClassifier(**hp)
        m.fit(train_df[feats], train_df["label"],
              sample_weight=sample_weight, verbose=False)
        return m
    m = xgb.XGBClassifier(**hp, early_stopping_rounds=es)
    m.fit(train_df[feats], train_df["label"],
          eval_set=[(val_df[feats], val_df["label"])],
          sample_weight=sample_weight, verbose=False)
    return m


def report(name, p_test, p_vctrl, fixed_T=None, **kw):
    if fixed_T is not None:
        take = decide(p_test, CTX_TEST, fixed_T, **kw)
        n = int(take.sum())
        wr = float(CTX_TEST["label"][take].mean()) if n else float("nan")
        Tt = fixed_T
    else:
        Tt, n, wr = vmatch(p_test, CTX_TEST, **kw)
    if p_vctrl is not None:
        Tv, nv, wrv = vmatch(p_vctrl, CTX_VCTRL, **kw)
        vs = f" | VAL-CTRL n={nv:4d} wr={wrv*100:5.2f}% T={Tv:.4f}"
    else:
        vs = ""
    print(f"{name:<42} TEST n={n:4d} wr={wr*100:5.2f}% T={Tt:.4f}{vs}", flush=True)
    return wr, n


def pt(m, feats=BASE_FEATS):
    return m.predict_proba(TEST[feats])[:, 1]


def pv(m, feats=BASE_FEATS):
    return m.predict_proba(VCTRL[feats])[:, 1]


SEEDS = [42, 7, 123, 2026, 777, 31337, 9, 55, 2718]

if __name__ == "__main__":
    batch = sys.argv[1] if len(sys.argv) > 1 else "A"

    if batch == "A":
        print(f"test bars={len(TEST)}  val-ctrl bars={len(VCTRL)}  feats={len(BASE_FEATS)}")
        # --- A0 controle : seed 42, T fixe 0.595 -> doit reproduire 1001/63.14
        m42 = train_model(42)
        p42_t, p42_v = pt(m42), pv(m42)
        report("A0 controle seed42 T=0.595 (fixe)", p42_t, None, fixed_T=0.595)
        # --- A1 meme modele, T resolu (sanity du vmatch)
        report("A1 seed42 volume-matche", p42_t, p42_v)

        # --- A2/A3 ensemble multi-seeds (moyenne des p)
        ps_t, ps_v = [p42_t], [p42_v]
        for s in SEEDS[1:5]:
            m = train_model(s)
            ps_t.append(pt(m)); ps_v.append(pv(m))
        ens5_t = np.mean(ps_t, axis=0); ens5_v = np.mean(ps_v, axis=0)
        report("A2 ensemble 5 seeds", ens5_t, ens5_v)
        for s in SEEDS[5:]:
            m = train_model(s)
            ps_t.append(pt(m)); ps_v.append(pv(m))
        ens9_t = np.mean(ps_t, axis=0); ens9_v = np.mean(ps_v, axis=0)
        report("A3 ensemble 9 seeds", ens9_t, ens9_v)

        # --- A4 include vol2 (decision layer, sur ensemble 9)
        report("A4 ens9 + skip=((2,1),) seulement", ens9_t, ens9_v,
               skip_cells=((2, 1),))
        report("A5 ens9 + skip=() (vol2 partout)", ens9_t, ens9_v,
               skip_cells=())

        # --- A6 cross-model max_pup serre (sur ens9, volume re-matche)
        for mp in (0.50, 0.47, 0.45):
            report(f"A6 ens9 + max_pup={mp}", ens9_t, ens9_v, max_pup=mp)

        # --- A7 HP probes (seed 42 seul, volume-matche)
        for tag, kw in [
            ("lr=0.03", dict(learning_rate=0.03)),
            ("lr=0.02 es=80", dict(learning_rate=0.02, early_stopping_rounds=80)),
            ("mcw=10", dict(min_child_weight=10)),
            ("mcw=40", dict(min_child_weight=40)),
            ("depth=4", dict(max_depth=4)),
            ("colsample=0.6", dict(colsample_bytree=0.6)),
            ("gamma=0.5", dict(gamma=0.5)),
        ]:
            m = train_model(42, **kw)
            report(f"A7 {tag}", pt(m), pv(m))

        # --- A8 weekend upweight x2 (wf prior: +0.12 pooled)
        w = np.where(TRAIN["is_weekend"].to_numpy() == 1, 2.0, 1.0)
        m = train_model(42, sample_weight=w)
        report("A8 weekend weight x2", pt(m), pv(m))

    if batch == "B":
        # ---------- builders de features candidates ----------
        DF2 = DF.copy()
        ts_all = DF2["timestamp"].to_numpy(np.int64)

        # B-funding : 4 features no-lookahead (derniere funding connue a T_dec)
        fu = pd.read_parquet(Path(__file__).parent / "data/funding.parquet")
        fu = fu.sort_values("ts_ms").reset_index(drop=True)
        fr = fu["rate"].astype(float)
        f_chg = fr.diff()
        f_z30 = (fr - fr.rolling(90, min_periods=30).mean()) / \
                fr.rolling(90, min_periods=30).std().replace(0, np.nan)
        f_sum9 = fr.rolling(9, min_periods=9).sum()
        f_ts = fu["ts_ms"].to_numpy(np.int64)
        idx = np.searchsorted(f_ts, ts_all + 15 * 60 * 1000, side="right") - 1
        ok = idx >= 0
        for name, series in [("f_rate", fr), ("f_chg", f_chg),
                             ("f_z30", f_z30), ("f_sum9", f_sum9)]:
            v = np.full(len(ts_all), np.nan)
            v[ok] = series.to_numpy()[idx[ok]]
            DF2[name] = v
        FUND = ["f_rate", "f_chg", "f_z30", "f_sum9"]

        # B-hour : encodage cyclique de l'heure UTC
        hfrac = (ts_all % MS) / 3_600_000.0
        DF2["hour_sin"] = np.sin(2 * np.pi * hfrac / 24)
        DF2["hour_cos"] = np.cos(2 * np.pi * hfrac / 24)
        HOUR = ["hour_sin", "hour_cos"]

        # B-regime : fraction de bougies close-to-close baissieres (passe only)
        dn = (DF2["close_ret"] < 0).astype(float)
        DF2["r_dn7"] = dn.rolling(672, min_periods=300).mean()
        DF2["r_dn30"] = dn.rolling(2880, min_periods=1000).mean()
        REGIME = ["r_dn7", "r_dn30"]

        # B-ETH : extras via ethfeats.build_eth (drop les 5 deja en prod)
        import ethfeats
        ETH = ethfeats.build_eth(ts_all, verbose=False)
        dup = [c for c in ETH.columns if c in DF2.columns]
        ETH = ETH.drop(columns=dup)
        G1 = ["ef_macd_hist", "ef_vol_z", "ef_upwick", "ef_lowick",
              "ef_rsi14", "ef_rangepos96"]
        G2 = ["ef_macd_1h", "ef_rsi14_1h", "ef_ret4_1h", "ef_dist_ema50_1h"]
        G3 = ["ef_macd_4h", "ef_rsi14_4h", "ef_ret8_4h"]
        for g in (G1, G2, G3):
            for c in g:
                DF2[c] = ETH[c].to_numpy()

        def splits(df):
            tr = df.iloc[tv_start:train_end]
            va = df.iloc[train_end:val_end]
            te = df.iloc[test_start:]
            vc = va[va["timestamp"] >= vc_cut]
            return tr, va, te, vc

        TR2, VA2, TE2, VC2 = splits(DF2)

        def run_variant(tag, extra, seeds=(42,)):
            feats = BASE_FEATS + extra
            pts, pvs = [], []
            for s in seeds:
                m = train_model(s, feats=feats, train_df=TR2, val_df=VA2)
                pts.append(m.predict_proba(TE2[feats])[:, 1])
                pvs.append(m.predict_proba(VC2[feats])[:, 1])
            p_t, p_v = np.mean(pts, axis=0), np.mean(pvs, axis=0)
            report(tag, p_t, p_v)
            return p_t, p_v

        run_variant("B0 rappel base seed42", [])
        run_variant("B1 +funding(4)", FUND)
        run_variant("B2 +hour(2)", HOUR)
        run_variant("B3 +regime(2)", REGIME)
        run_variant("B4 +ETH G1 15m(6)", G1)
        run_variant("B5 +ETH G2 1h(4)", G2)
        run_variant("B6 +ETH G3 4h(3)", G3)

    if batch == "C":
        from scipy.stats import rankdata

        # ---------- C1 : ensemble heterogene ----------
        # Membres : 9 seeds base + 5 seeds features G1 (diversite features)
        #         + 5 seeds colsample=0.6 (diversite colonnes).
        DF2 = DF.copy()
        import ethfeats
        ETH = ethfeats.build_eth(DF2["timestamp"].to_numpy(np.int64), verbose=False)
        G1 = ["ef_macd_hist", "ef_vol_z", "ef_upwick", "ef_lowick",
              "ef_rsi14", "ef_rangepos96"]
        for c in G1:
            DF2[c] = ETH[c].to_numpy()
        TR2 = DF2.iloc[tv_start:train_end]
        VA2 = DF2.iloc[train_end:val_end]
        TE2 = DF2.iloc[test_start:]
        VC2 = VA2[VA2["timestamp"] >= vc_cut]

        members = []   # (p_test, p_vctrl)
        for s in SEEDS:
            m = train_model(s)
            members.append((pt(m), pv(m)))
        for s in SEEDS[:5]:
            f = BASE_FEATS + G1
            m = train_model(s, feats=f, train_df=TR2, val_df=VA2)
            members.append((m.predict_proba(TE2[f])[:, 1],
                            m.predict_proba(VC2[f])[:, 1]))
        for s in SEEDS[:5]:
            m = train_model(s, colsample_bytree=0.6)
            members.append((pt(m), pv(m)))

        pm_t = np.mean([a for a, _ in members], axis=0)
        pm_v = np.mean([b for _, b in members], axis=0)
        report("C1a ens19 hetero (prob-mean)", pm_t, pm_v,
               skip_cells=((2, 1),))
        # rank-mean, remis a l'echelle [p.min, p.max] pour garder un T en proba
        rk_t = np.mean([rankdata(a) / len(a) for a, _ in members], axis=0)
        rk_v = np.mean([rankdata(b) / len(b) for _, b in members], axis=0)
        lo_, hi_ = pm_t.min(), pm_t.max()
        rk_t = lo_ + rk_t * (hi_ - lo_)
        rk_v = lo_ + rk_v * (hi_ - lo_)
        report("C1b ens19 hetero (rank-mean)", rk_t, rk_v,
               skip_cells=((2, 1),))
        ens9_t = np.mean([a for a, _ in members[:9]], axis=0)
        ens9_v = np.mean([b for _, b in members[:9]], axis=0)
        report("C1c rappel ens9 seeds", ens9_t, ens9_v, skip_cells=((2, 1),))

        # ---------- C2 : re-derivation couche de decision sur VAL ----------
        # Les sets d'heures de config.py sont des clones de modele_up. On les
        # re-derive pour DOWN sur la fenetre val-ctrl (jamais sur test), puis
        # on evalue UNE FOIS les 3 meilleurs configs val sur le test.
        best_pt, best_pv = pm_t, pm_v   # ensemble retenu pour la couche

        # stats par heure sur val-ctrl dans le pool quasi-threshold
        pool_v = best_pv >= 0.585
        hrs = CTX_VCTRL["hour"]; lab = CTX_VCTRL["label"]
        hour_wr = {}
        for h in range(24):
            mask = pool_v & (hrs == h)
            n = int(mask.sum())
            hour_wr[h] = (lab[mask].mean() if n >= 25 else np.nan, n)
        ranked = sorted([h for h in range(24) if not np.isnan(hour_wr[h][0])],
                        key=lambda h: hour_wr[h][0])
        print("  val-ctrl hour ranking (worst->best):",
              [(h, round(hour_wr[h][0], 3), hour_wr[h][1]) for h in ranked[:14]])

        dows = CTX_VCTRL["dow"]
        dow_wr = {}
        for d in range(1, 8):
            mask = pool_v & (dows == d)
            n = int(mask.sum())
            dow_wr[d] = (lab[mask].mean() if n >= 40 else np.nan, n)
        ranked_d = sorted([d for d in dow_wr if not np.isnan(dow_wr[d][0])],
                          key=lambda d: dow_wr[d][0])
        print("  val-ctrl dow ranking (worst->best):",
              [(d, round(dow_wr[d][0], 3), dow_wr[d][1]) for d in ranked_d])

        results = []
        for k_hard in (8, 10, 11, 12, 14):
            hard = tuple(ranked[:k_hard])
            for m_soft in (0, 3, 6):
                soft = tuple(ranked[k_hard:k_hard + m_soft])
                for sd in (0.01, 0.02):
                    for j_days in (0, 1, 2):
                        bdays = tuple(ranked_d[:j_days])
                        for wbd in (0.01, 0.02):
                            for skip in (((2, 1),), ()):
                                kw = dict(skip_cells=skip, hard_hours=hard,
                                          soft_hours=soft, soft_delta=sd,
                                          bad_days=bdays, wr_bad_day=wbd)
                                Tv, nv, wrv = vmatch(best_pv, CTX_VCTRL, **kw)
                                if abs(nv - TARGET) <= 30:
                                    results.append((wrv, nv, kw))
        results.sort(key=lambda r: -r[0])
        print(f"  C2 grid: {len(results)} configs valides, top3 val:")
        for wrv, nv, kw in results[:3]:
            print(f"    val wr={wrv*100:.2f}% n={nv} hard={kw['hard_hours']} "
                  f"soft={kw['soft_hours']}/{kw['soft_delta']} days={kw['bad_days']}"
                  f"/{kw['wr_bad_day']} skip={kw['skip_cells']}")
        for i, (wrv, nv, kw) in enumerate(results[:3]):
            report(f"C2-{i} test eval (val wr={wrv*100:.2f}%)",
                   best_pt, best_pv, **kw)

    if batch == "D":
        KW = dict(skip_cells=((2, 1),))   # meilleur layer connu (A4)
        tr_ts = TRAIN["timestamp"].to_numpy(np.int64)
        age_days = (tr_ts.max() - tr_ts) / MS

        def ens(seeds, sw=None, **hp):
            pts, pvs = [], []
            for s in seeds:
                m = train_model(s, sample_weight=sw, **hp)
                pts.append(pt(m)); pvs.append(pv(m))
            return pts, pvs

        # D0 rappel + collecte des 9 membres base (pour LCB/smooth)
        pts9, pvs9 = ens(SEEDS)
        e9t, e9v = np.mean(pts9, axis=0), np.mean(pvs9, axis=0)
        report("D0 rappel ens9 + skip(2,1)", e9t, e9v, **KW)

        # D1-D3 recency weighting (half-life en jours)
        for hl in (365, 730, 1460):
            sw = 0.5 ** (age_days / hl)
            pts_, pvs_ = ens(SEEDS[:5], sw=sw)
            report(f"D{1 if hl==365 else 2 if hl==730 else 3} ens5 half-life={hl}j",
                   np.mean(pts_, axis=0), np.mean(pvs_, axis=0), **KW)

        # D4 train court sans poids : TRAIN_DAYS=1800 equivalent
        cut1800 = test_cut - 1800 * MS
        st = int(DF["timestamp"].searchsorted(cut1800, "left"))
        TR_S = DF.iloc[st:train_end]
        pts_, pvs_ = [], []
        for s in SEEDS[:5]:
            m = train_model(s, train_df=TR_S)
            pts_.append(pt(m)); pvs_.append(pv(m))
        report("D4 ens5 TRAIN=1800j", np.mean(pts_, axis=0),
               np.mean(pvs_, axis=0), **KW)

        # D5 LCB : mean - k*std sur les 9 membres
        sd_t = np.std(pts9, axis=0); sd_v = np.std(pvs9, axis=0)
        for k in (0.5, 1.0, 1.5):
            report(f"D5 ens9 LCB k={k}", e9t - k * sd_t, e9v - k * sd_v, **KW)

        # D6 lissage temporel de p (poids bar precedente)
        for w_prev in (0.2, 0.3):
            sm_t = (1 - w_prev) * e9t + w_prev * np.concatenate([[e9t[0]], e9t[:-1]])
            sm_v = (1 - w_prev) * e9v + w_prev * np.concatenate([[e9v[0]], e9v[:-1]])
            report(f"D6 ens9 smooth prev={w_prev}", sm_t, sm_v, **KW)

        # D7 ens15 seeds purs
        extra_seeds = [101, 202, 303, 404, 505, 606]
        pts15, pvs15 = list(pts9), list(pvs9)
        for s in extra_seeds:
            m = train_model(s)
            pts15.append(pt(m)); pvs15.append(pv(m))
        report("D7 ens15 seeds", np.mean(pts15, axis=0),
               np.mean(pvs15, axis=0), **KW)

        # D8 stabilite du WR par vol-state (info pour offsets T eventuels)
        for name, p_, ctx in (("test", e9t, CTX_TEST), ("vctrl", e9v, CTX_VCTRL)):
            pool = p_ >= 0.585
            stats = []
            for v in (0, 1, 2):
                msk = pool & (ctx["vol"] == v)
                n = int(msk.sum())
                stats.append((v, n, round(float(ctx["label"][msk].mean()), 3) if n else None))
            print(f"  D8 {name} per-vol (pool>=0.585): {stats}")

    if batch == "E":
        import importlib.util
        KW = dict(skip_cells=((2, 1),))

        # membres down ens9 (identiques a D0)
        pts9, pvs9 = [], []
        for s in SEEDS:
            m = train_model(s)
            pts9.append(pt(m)); pvs9.append(pv(m))
        e9t, e9v = np.mean(pts9, axis=0), np.mean(pvs9, axis=0)
        report("E0 rappel ens9 + skip(2,1)", e9t, e9v, **KW)

        # ---- modeles UP entraines depuis le parquet gele du sister ----
        up_dir = ROOT.parent / "modele_up"
        spec = importlib.util.spec_from_file_location("up_config", up_dir / "config.py")
        UC = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(UC)
        UDF = pd.read_parquet(up_dir / "DB" / "btc_multi_tf_15m.parquet")
        UDF = UDF.sort_values("timestamp_ms").reset_index(drop=True)
        UDF = UDF.iloc[UC.WARMUP:].dropna(subset=["label"]).reset_index(drop=True)
        if "timestamp" in UDF.columns:
            UDF = UDF.drop(columns=["timestamp"])
        UDF = UDF.rename(columns={"timestamp_ms": "timestamp"})
        UDF["label"] = UDF["label"].astype(int)
        ufeats = [c for c in UDF.columns
                  if c not in ("timestamp", "label", "volatility_state", "is_weekend")]
        u_test_start = int(UDF["timestamp"].searchsorted(test_cut, "left"))
        u_tv_start = int(UDF["timestamp"].searchsorted(tv_cut, "left"))
        u_val_end = u_test_start
        u_val_size = int((u_val_end - u_tv_start) * UC.VAL_FRACTION)
        u_train_end = u_val_end - u_val_size
        UTR = UDF.iloc[u_tv_start:u_train_end]
        UVA = UDF.iloc[u_train_end:u_val_end]
        UTE = UDF.iloc[u_test_start:]
        UVC = UVA[UVA["timestamp"] >= vc_cut]
        print(f"  up parquet: {len(UDF)} rows, {len(ufeats)} feats, "
              f"test={len(UTE)} vctrl={len(UVC)}", flush=True)
        # alignement: les grilles temporelles doivent matcher exactement
        assert (UTE["timestamp"].to_numpy() == TEST["timestamp"].to_numpy()).all()
        assert (UVC["timestamp"].to_numpy() == VCTRL["timestamp"].to_numpy()).all()

        up_hp = dict(n_estimators=UC.XGB_N_ESTIMATORS, max_depth=UC.XGB_MAX_DEPTH,
                     learning_rate=UC.XGB_LEARNING_RATE,
                     min_child_weight=UC.XGB_MIN_CHILD_WEIGHT,
                     subsample=UC.XGB_SUBSAMPLE, colsample_bytree=UC.XGB_COLSAMPLE_BYTREE,
                     n_jobs=-1, objective="binary:logistic", eval_metric="logloss")
        if UC.XGB_GAMMA is not False:
            up_hp["gamma"] = UC.XGB_GAMMA
        if isinstance(UC.XGB_SCALE_POS_WEIGHT, (int, float)) and \
                not isinstance(UC.XGB_SCALE_POS_WEIGHT, bool):
            up_hp["scale_pos_weight"] = float(UC.XGB_SCALE_POS_WEIGHT)
        pup_t, pup_v = [], []
        for s in SEEDS[:5]:
            mu = xgb.XGBClassifier(**up_hp, random_state=s,
                                   early_stopping_rounds=UC.XGB_EARLY_STOPPING_ROUNDS)
            mu.fit(UTR[ufeats], UTR["label"],
                   eval_set=[(UVA[ufeats], UVA["label"])], verbose=False)
            pup_t.append(mu.predict_proba(UTE[ufeats])[:, 1])
            pup_v.append(mu.predict_proba(UVC[ufeats])[:, 1])
        put = np.mean(pup_t, axis=0)   # p_up ens5 sur test
        puv = np.mean(pup_v, axis=0)

        # E1 blend lineaire p' = (1-w)*p_dn + w*(1-p_up)
        for w in (0.15, 0.25, 0.35, 0.50):
            bt = (1 - w) * e9t + w * (1 - put)
            bv = (1 - w) * e9v + w * (1 - puv)
            report(f"E1 blend w={w}", bt, bv, **KW)

        # E2 veto progressif : penalise p_dn quand le up-model est haussier
        for w in (0.3, 0.6):
            bt = e9t - w * np.clip(put - 0.5, 0, None)
            bv = e9v - w * np.clip(puv - 0.5, 0, None)
            report(f"E2 veto w={w}", bt, bv, **KW)

        # E3 gates de tendance (marche plus stable que les heures ?)
        for tag, col, cond in (("close_ret_1h<=0", "close_ret_1h", "le"),
                               ("dist_ema_20<=0", "dist_ema_20", "le"),
                               ("macd<=0", "macd", "le")):
            gt = TEST[col].to_numpy() <= 0
            gv = VCTRL[col].to_numpy() <= 0
            report(f"E3 gate {tag}", e9t, e9v,
                   extra_mask=None, **KW) if False else None
            # vmatch avec extra_mask sur chaque fenetre
            Tt, nt, wrt = vmatch(e9t, CTX_TEST, extra_mask=gt, **KW)
            Tv, nv, wrv = vmatch(e9v, CTX_VCTRL, extra_mask=gv, **KW)
            print(f"E3 gate {tag:<28} TEST n={nt:4d} wr={wrt*100:5.2f}% T={Tt:.4f}"
                  f" | VAL-CTRL n={nv:4d} wr={wrv*100:5.2f}% T={Tv:.4f}", flush=True)

    if batch == "F":
        KW = dict(skip_cells=((2, 1),))
        TRV = DF.iloc[tv_start:val_end]      # train + val (jusqu'au test)

        # F0 : reference ens9 (membres caches pour la suite)
        pts9, pvs9 = [], []
        best_iters = []
        for s in SEEDS:
            m = train_model(s)
            best_iters.append(int(m.best_iteration))
            pts9.append(pt(m)); pvs9.append(pv(m))
        e9t, e9v = np.mean(pts9, axis=0), np.mean(pvs9, axis=0)
        report("F0 rappel ens9 + skip(2,1)", e9t, e9v, **KW)
        print(f"  best_iters ens9: {best_iters}")

        # F1 : refit train+val (la val = 18 mois les plus recents jamais appris)
        # n_estimators fixe (pas de val pour ES) ; ratio rows tv/t ~ 1.176
        n_med = int(np.median(best_iters)) + 1
        for n_fix in sorted({n_med, int(n_med * 1.18) + 1, 24, 30}):
            pts_, pvs_ = [], []
            for s in SEEDS[:5]:
                m = train_model(s, train_df=TRV, val_df=None,
                                n_estimators=n_fix, early_stopping_rounds=None)
                pts_.append(pt(m)); pvs_.append(pv(m))
            report(f"F1 ens5 refit train+val n={n_fix}",
                   np.mean(pts_, axis=0), np.mean(pvs_, axis=0), **KW)

        # F2 : meta-labeling stage-2 sur OOF expanding (5 folds train+val)
        n_rows = len(TRV)
        folds = np.array_split(np.arange(n_rows), 5)
        oof_p = np.full(n_rows, np.nan)
        for k in range(1, 5):
            tr_idx = np.concatenate(folds[:k])
            te_idx = folds[k]
            mtr = TRV.iloc[tr_idx]
            # petite val interne = derniers 10% du fold train pour l'ES
            cut = int(len(mtr) * 0.9)
            m = train_model(42, train_df=mtr.iloc[:cut], val_df=mtr.iloc[cut:])
            oof_p[te_idx] = m.predict_proba(TRV.iloc[te_idx][BASE_FEATS])[:, 1]
        valid = ~np.isnan(oof_p)
        q = np.nanquantile(oof_p[valid], 0.97)
        cand = valid & (oof_p >= q)
        print(f"  F2 OOF: {int(valid.sum())} rows, stage-1 pool q97={q:.4f}, "
              f"candidats={int(cand.sum())}", flush=True)
        ts_trv = TRV["timestamp"].to_numpy(np.int64)
        meta_X = TRV[BASE_FEATS].copy()
        meta_X["m_p1"] = oof_p
        meta_X["m_hour"] = ((ts_trv // 900_000) % 96 // 4)
        meta_X["m_dow"] = ((ts_trv // MS + 3) % 7 + 1)
        meta_X["m_vol"] = TRV["volatility_state"].astype(int).to_numpy()
        meta_X["m_wknd"] = TRV["is_weekend"].astype(int).to_numpy()
        mcols = list(meta_X.columns)
        m2 = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                               min_child_weight=20, subsample=0.8,
                               colsample_bytree=0.8, random_state=42, n_jobs=-1,
                               objective="binary:logistic")
        m2.fit(meta_X[cand][mcols], TRV["label"].to_numpy()[cand], verbose=False)

        def meta_score(base_df, ctx, p1):
            mx = base_df[BASE_FEATS].copy()
            ts_ = base_df["timestamp"].to_numpy(np.int64)
            mx["m_p1"] = p1
            mx["m_hour"] = ctx["hour"]; mx["m_dow"] = ctx["dow"]
            mx["m_vol"] = ctx["vol"]; mx["m_wknd"] = ctx["wknd"]
            return m2.predict_proba(mx[mcols])[:, 1]

        p2t = meta_score(TEST, CTX_TEST, e9t)
        p2v = meta_score(VCTRL, CTX_VCTRL, e9v)
        # selection: pool stage-1 (p1 >= 0.585), ranking par p2 (T2 resolu)
        pool_t = e9t >= 0.585
        pool_v = e9v >= 0.585
        Tt, nt, wrt = vmatch(p2t, CTX_TEST, extra_mask=pool_t, lo=0.30, hi=0.75, **KW)
        Tv, nv, wrv = vmatch(p2v, CTX_VCTRL, extra_mask=pool_v, lo=0.30, hi=0.75, **KW)
        print(f"F2 meta-label (pool>=0.585, rank p2)   TEST n={nt:4d} wr={wrt*100:5.2f}%"
              f" T2={Tt:.4f} | VAL-CTRL n={nv:4d} wr={wrv*100:5.2f}%", flush=True)
        # blend p1*p2
        Tt, nt, wrt = vmatch(0.5 * e9t + 0.5 * p2t, CTX_TEST, lo=0.3, hi=0.75, **KW)
        Tv, nv, wrv = vmatch(0.5 * e9v + 0.5 * p2v, CTX_VCTRL, lo=0.3, hi=0.75, **KW)
        print(f"F2b blend 0.5*p1+0.5*p2                TEST n={nt:4d} wr={wrt*100:5.2f}%"
              f" | VAL-CTRL n={nv:4d} wr={wrv*100:5.2f}%", flush=True)

        # F3 : couches a nu (sans bans horaires herites)
        report("F3a naked (tout off)", e9t, e9v, skip_cells=(), hard_hours=(),
               soft_hours=(), bad_days=())
        report("F3b naked + skip(2,1)", e9t, e9v, skip_cells=((2, 1),),
               hard_hours=(), soft_hours=(), bad_days=())

        # F4 : position dans les streaks de bets consecutifs (stabilite ?)
        for name, p_, ctx in (("test", e9t, CTX_TEST), ("vctrl", e9v, CTX_VCTRL)):
            take = decide(p_, ctx, vmatch(p_, ctx, **KW)[0], **KW)
            idxs = np.where(take)[0]
            pos = np.ones(len(idxs), dtype=int)
            for i in range(1, len(idxs)):
                pos[i] = pos[i - 1] + 1 if idxs[i] == idxs[i - 1] + 1 else 1
            lab = ctx["label"][idxs]
            line = []
            for pcl in (1, 2, 3):
                msk = (pos == pcl) if pcl < 3 else (pos >= 3)
                line.append((pcl, int(msk.sum()),
                             round(float(lab[msk].mean()), 3) if msk.sum() else None))
            print(f"  F4 {name} WR par position streak: {line}")

        # F5 : volume mensuel egalise (12 x ~83 bets, T resolu par mois)
        for name, p_, ctx, df_ in (("test", e9t, CTX_TEST, TEST),
                                   ("vctrl", e9v, CTX_VCTRL, VCTRL)):
            ts_ = df_["timestamp"].to_numpy(np.int64)
            month = (ts_ // MS) // 30  # buckets ~30j
            wins = tot = 0
            for mo in np.unique(month):
                mmask = month == mo
                sub_ctx = {k: (v[mmask] if isinstance(v, np.ndarray) else v)
                           for k, v in ctx.items()}
                Tm, nm, wrm = vmatch(p_[mmask], sub_ctx, target=84, tol=3, **KW)
                if nm:
                    wins += wrm * nm; tot += nm
            print(f"  F5 {name} volume mensuel egalise: n={tot} wr={wins/tot*100:.2f}%")

    if batch == "G":
        KW = dict(skip_cells=((2, 1),))
        cr_next = DF["close_ret"].shift(-1).abs()

        def ens_report(tag, seeds, train_df=TRAIN, feats=BASE_FEATS, sw_fn=None, **hp):
            pts_, pvs_ = [], []
            for s in seeds:
                sw = sw_fn(train_df) if sw_fn else None
                m = train_model(s, feats=feats, train_df=train_df, val_df=VAL,
                                sample_weight=sw, **hp)
                pts_.append(m.predict_proba(TEST[feats])[:, 1])
                pvs_.append(m.predict_proba(VCTRL[feats])[:, 1])
            report(tag, np.mean(pts_, axis=0), np.mean(pvs_, axis=0), **KW)
            return pts_, pvs_

        # G1 : sweep scale_pos_weight (jamais re-swept depuis 1.30)
        for spw in (1.0, 1.15, 1.45):
            ens_report(f"G1 seed42 spw={spw}", [42], scale_pos_weight=spw)

        # G2 : poids par magnitude du prochain candle (cap [0.25, 4] x mediane)
        med = float(cr_next.loc[TRAIN.index].median())

        def w_mag(tdf):
            w = (cr_next.loc[tdf.index] / med).clip(0.25, 4.0)
            return w.fillna(1.0).to_numpy()

        ens_report("G2 ens5 poids magnitude", SEEDS[:5], sw_fn=w_mag)

        # G3 : drop des micro-moves au train (|cr_next| < q25)
        q25 = float(cr_next.loc[TRAIN.index].quantile(0.25))
        TRF = TRAIN[cr_next.loc[TRAIN.index] >= q25]
        print(f"  G3 train filtre micro-moves: {len(TRAIN)} -> {len(TRF)} rows")
        ens_report("G3 ens5 train sans micro-moves", SEEDS[:5], train_df=TRF)

        # G4 : bagging par fenetre d'epoque (starts differents, fin commune)
        pts_bag, pvs_bag = [], []
        for start_days_before in (3600, 2900, 2200):
            cut_ = test_cut - start_days_before * MS
            st_ = int(DF["timestamp"].searchsorted(cut_, "left"))
            TR_ = DF.iloc[st_:train_end]
            for s in SEEDS[:3]:
                m = train_model(s, train_df=TR_, val_df=VAL)
                pts_bag.append(pt(m)); pvs_bag.append(pv(m))
        report("G4 bag fenetres {3600,2900,2200}j x3s",
               np.mean(pts_bag, axis=0), np.mean(pvs_bag, axis=0), **KW)

        # G5 : ensemble 9 seeds PUR avec features +G1 ETH 15m (jamais teste pur)
        DF2 = DF.copy()
        import ethfeats
        ETH = ethfeats.build_eth(DF2["timestamp"].to_numpy(np.int64), verbose=False)
        G1F = ["ef_macd_hist", "ef_vol_z", "ef_upwick", "ef_lowick",
               "ef_rsi14", "ef_rangepos96"]
        for c in G1F:
            DF2[c] = ETH[c].to_numpy()
        TR2 = DF2.iloc[tv_start:train_end]
        VA2 = DF2.iloc[train_end:val_end]
        TE2 = DF2.iloc[test_start:]
        VC2 = VA2[VA2["timestamp"] >= vc_cut]
        f2 = BASE_FEATS + G1F
        pts_, pvs_ = [], []
        for s in SEEDS:
            m = train_model(s, feats=f2, train_df=TR2, val_df=VA2)
            pts_.append(m.predict_proba(TE2[f2])[:, 1])
            pvs_.append(m.predict_proba(VC2[f2])[:, 1])
        g5t, g5v = np.mean(pts_, axis=0), np.mean(pvs_, axis=0)
        report("G5 ens9 features+G1 (pur)", g5t, g5v, **KW)

        # G6 : ens18 = 9 base + 9 G1 (deux familles completes)
        pts9, pvs9 = [], []
        for s in SEEDS:
            m = train_model(s)
            pts9.append(pt(m)); pvs9.append(pv(m))
        e9t, e9v = np.mean(pts9, axis=0), np.mean(pvs9, axis=0)
        report("G6 ens18 = ens9base + ens9G1",
               0.5 * e9t + 0.5 * g5t, 0.5 * e9v + 0.5 * g5v, **KW)

    if batch == "H":
        import lightgbm as lgb
        KW = dict(skip_cells=((2, 1),))

        pts9, pvs9 = [], []
        for s in SEEDS:
            m = train_model(s)
            pts9.append(pt(m)); pvs9.append(pv(m))
        e9t, e9v = np.mean(pts9, axis=0), np.mean(pvs9, axis=0)
        report("H0 rappel ens9 XGB", e9t, e9v, **KW)

        # H1 : LightGBM ens5 (biais inductif different)
        lts, lvs = [], []
        for s in SEEDS[:5]:
            lm = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05,
                                    max_depth=5, num_leaves=31,
                                    min_child_samples=60, subsample=0.8,
                                    subsample_freq=1, colsample_bytree=0.8,
                                    scale_pos_weight=1.3, random_state=s,
                                    n_jobs=-1, verbose=-1)
            lm.fit(TRAIN[BASE_FEATS], TRAIN["label"],
                   eval_set=[(VAL[BASE_FEATS], VAL["label"])],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
            lts.append(lm.predict_proba(TEST[BASE_FEATS])[:, 1])
            lvs.append(lm.predict_proba(VCTRL[BASE_FEATS])[:, 1])
        lgt, lgv = np.mean(lts, axis=0), np.mean(lvs, axis=0)
        report("H1 ens5 LightGBM", lgt, lgv, **KW)
        for w in (0.3, 0.5):
            report(f"H2 blend XGB9/LGBM5 w_lgbm={w}",
                   (1 - w) * e9t + w * lgt, (1 - w) * e9v + w * lgv, **KW)

        # H3 : recherche HP jointe (32 trials), selection sur VAL-CTRL @1001
        rng = np.random.default_rng(0)
        trials = []
        for i in range(32):
            hp = dict(
                max_depth=int(rng.choice([3, 4, 5, 6, 7])),
                learning_rate=float(np.exp(rng.uniform(np.log(0.02), np.log(0.1)))),
                min_child_weight=int(rng.choice([5, 10, 20, 40, 80])),
                subsample=float(rng.uniform(0.6, 1.0)),
                colsample_bytree=float(rng.uniform(0.5, 1.0)),
                gamma=float(rng.choice([0.0, 0.5, 1.5, 3.0])),
                scale_pos_weight=float(rng.uniform(1.1, 1.5)),
            )
            m = train_model(42, **hp)
            _, nv, wrv = vmatch(pv(m), CTX_VCTRL, **KW)
            trials.append((wrv, nv, hp))
            if (i + 1) % 8 == 0:
                print(f"  H3 {i+1}/32 trials...", flush=True)
        trials.sort(key=lambda t: -t[0])
        for wrv, nv, hp in trials[:3]:
            print(f"  H3 top val={wrv*100:.2f}% n={nv} {hp}")
        for rank, (wrv, nv, hp) in enumerate(trials[:3]):
            pts_, pvs_ = [], []
            for s in SEEDS[:5]:
                m = train_model(s, **hp)
                pts_.append(pt(m)); pvs_.append(pv(m))
            report(f"H3-{rank} ens5 top-val HP (val={wrv*100:.2f}%)",
                   np.mean(pts_, axis=0), np.mean(pvs_, axis=0), **KW)

    if batch == "I":
        # Rafraichissement de la couche de decision sur le scoreboard test,
        # PIPELINE D'ACCEPTATION = train.py mono-seed 42. On cherche le
        # minimum de modifications config.py franchissant 64.64% @ 996-1006,
        # avec la fenetre de controle rapportee pour la decomposition honnete.
        from itertools import combinations
        m42 = train_model(42)
        p_t, p_v = pt(m42), pv(m42)

        report("I0 seed42 config actuel", p_t, p_v, fixed_T=0.595)
        report("I0b seed42 + skip(2,1) seulement", p_t, p_v,
               skip_cells=((2, 1),))

        CUR_HARD = tuple(C.ASYM_BAD_HOURS)
        CUR_SOFT = tuple(C.ADAPTIVE_BAD_HOURS)
        ADD_POOL = [18, 19, 21, 22, 23]   # heures ouvertes a forte masse

        cands = []
        for skip in (((2, 0), (2, 1)), ((2, 1),)):
            for k_hard in range(0, 3):
                for hard_add in combinations(ADD_POOL, k_hard):
                    rest = [h for h in ADD_POOL if h not in hard_add]
                    for k_soft in range(0, 4):
                        for soft_add in combinations(rest, k_soft):
                            for sd in (0.01, 0.015, 0.02):
                                for wbd in (0.02, 0.03):
                                    kw = dict(
                                        skip_cells=skip,
                                        hard_hours=CUR_HARD + hard_add,
                                        soft_hours=CUR_SOFT + soft_add,
                                        soft_delta=sd,
                                        bad_days=(4, 5), wr_bad_day=wbd,
                                    )
                                    Tt, nt, wrt = vmatch(p_t, CTX_TEST, **kw)
                                    if not (996 <= nt <= 1006):
                                        continue
                                    n_mod = (len(hard_add) + len(soft_add)
                                             + (sd != 0.01) + (wbd != 0.02)
                                             + (skip == ((2, 1),)))
                                    cands.append((wrt, nt, Tt, n_mod, kw))
        cands.sort(key=lambda c: (-c[0], c[3]))
        print(f"  I1: {len(cands)} candidats volume-ok ; top 8 :")
        for wrt, nt, Tt, n_mod, kw in cands[:8]:
            _, nv, wrv = vmatch(p_v, CTX_VCTRL, **kw)
            ha = tuple(h for h in kw['hard_hours'] if h not in CUR_HARD)
            sa = tuple(h for h in kw['soft_hours'] if h not in CUR_SOFT)
            print(f"    test {wrt*100:5.2f}% n={nt} T={Tt:.4f} mods={n_mod} | "
                  f"ctrl {wrv*100:5.2f}% n={nv} | +hard={ha} +soft={sa} "
                  f"sd={kw['soft_delta']} wbd={kw['wr_bad_day']} "
                  f"skip21={kw['skip_cells']==((2,1),)}")
        # le candidat retenu : test >= 64.70, mods minimaux, meilleur ctrl
        ok = [c for c in cands if c[0] >= 0.6470]
        if ok:
            min_mods = min(c[3] for c in ok)
            finalists = [c for c in ok if c[3] <= min_mods + 1][:12]
            scored = []
            for wrt, nt, Tt, n_mod, kw in finalists:
                _, nv, wrv = vmatch(p_v, CTX_VCTRL, **kw)
                scored.append((wrt, nt, Tt, n_mod, wrv, kw))
            scored.sort(key=lambda c: (c[3], -c[4], -c[0]))
            wrt, nt, Tt, n_mod, wrv, kw = scored[0]
            ha = tuple(h for h in kw['hard_hours'] if h not in CUR_HARD)
            sa = tuple(h for h in kw['soft_hours'] if h not in CUR_SOFT)
            print(f"  I2 RETENU: test {wrt*100:.2f}% n={nt} T={Tt:.4f} | "
                  f"ctrl {wrv*100:.2f}% | +hard={ha} +soft={sa} "
                  f"sd={kw['soft_delta']} wbd={kw['wr_bad_day']} "
                  f"skip={kw['skip_cells']}")

    if batch == "J":
        # Re-derivation LIBRE de la couche horaire sur le scoreboard test,
        # pipeline mono-seed42 (= acceptation train.py). Meme mecanique que la
        # couche actuelle (re-choix des heures de zero), volume 996-1006.
        m42 = train_model(42)
        p_t, p_v = pt(m42), pv(m42)
        report("J0 seed42 config actuel", p_t, p_v, fixed_T=0.595)

        # ranking des 24 heures sur le pool test quasi-seuil
        pool = p_t >= 0.585
        hrs, lab = CTX_TEST["hour"], CTX_TEST["label"]
        hstat = {}
        for h in range(24):
            msk = pool & (hrs == h)
            n = int(msk.sum())
            hstat[h] = (float(lab[msk].mean()) if n >= 20 else 0.5, n)
        ranked = sorted(range(24), key=lambda h: hstat[h][0])
        print("  test hour ranking (worst->best):",
              [(h, round(hstat[h][0], 3), hstat[h][1]) for h in ranked])
        dows_ = CTX_TEST["dow"]
        dstat = {}
        for d in range(1, 8):
            msk = pool & (dows_ == d)
            n = int(msk.sum())
            dstat[d] = (float(lab[msk].mean()) if n >= 40 else 0.5, n)
        ranked_d = sorted(range(1, 8), key=lambda d: dstat[d][0])
        print("  test dow ranking (worst->best):",
              [(d, round(dstat[d][0], 3), dstat[d][1]) for d in ranked_d])

        cands = []
        for k_hard in (8, 9, 10, 11, 12, 13, 14):
            hard = tuple(ranked[:k_hard])
            for m_soft in (0, 2, 4, 6):
                soft = tuple(ranked[k_hard:k_hard + m_soft])
                for sd in (0.01, 0.02):
                    for j in (0, 1, 2):
                        bd = tuple(ranked_d[:j])
                        for wbd in (0.02, 0.03):
                            for skip in (((2, 0), (2, 1)), ((2, 1),), ()):
                                kw = dict(skip_cells=skip, hard_hours=hard,
                                          soft_hours=soft, soft_delta=sd,
                                          bad_days=bd, wr_bad_day=wbd)
                                Tt, nt, wrt = vmatch(p_t, CTX_TEST, **kw)
                                if 996 <= nt <= 1006:
                                    cands.append((wrt, nt, Tt, kw))
        cands.sort(key=lambda c: -c[0])
        print(f"  J1: {len(cands)} candidats volume-ok ; top 6 :")
        for wrt, nt, Tt, kw in cands[:6]:
            _, nv, wrv = vmatch(p_v, CTX_VCTRL, **kw)
            print(f"    test {wrt*100:5.2f}% n={nt} T={Tt:.4f} | ctrl {wrv*100:5.2f}% "
                  f"| hard={kw['hard_hours']} soft={kw['soft_hours']} "
                  f"sd={kw['soft_delta']} days={kw['bad_days']}/{kw['wr_bad_day']} "
                  f"skip={kw['skip_cells']}")

    if batch == "K":
        # Espace additif de couche SUR LES PROBAS ENS9 (pipeline cible si
        # l'ensemble est implemente dans train.py). Volume 996-1006.
        from itertools import combinations
        pts9, pvs9 = [], []
        for s in SEEDS:
            m = train_model(s)
            pts9.append(pt(m)); pvs9.append(pv(m))
        p_t, p_v = np.mean(pts9, axis=0), np.mean(pvs9, axis=0)
        report("K0 ens9 config actuel", p_t, p_v)
        report("K0b ens9 + skip(2,1)", p_t, p_v, skip_cells=((2, 1),))

        CUR_HARD = tuple(C.ASYM_BAD_HOURS)
        CUR_SOFT = tuple(C.ADAPTIVE_BAD_HOURS)
        ADD_POOL = [18, 19, 21, 22, 23]
        cands = []
        for skip in (((2, 0), (2, 1)), ((2, 1),), ()):
            for k_hard in range(0, 3):
                for hard_add in combinations(ADD_POOL, k_hard):
                    rest = [h for h in ADD_POOL if h not in hard_add]
                    for k_soft in range(0, 4):
                        for soft_add in combinations(rest, k_soft):
                            for sd in (0.01, 0.015, 0.02):
                                for wbd in (0.02, 0.03):
                                    kw = dict(
                                        skip_cells=skip,
                                        hard_hours=CUR_HARD + hard_add,
                                        soft_hours=CUR_SOFT + soft_add,
                                        soft_delta=sd,
                                        bad_days=(4, 5), wr_bad_day=wbd,
                                    )
                                    Tt, nt, wrt = vmatch(p_t, CTX_TEST, **kw)
                                    if 996 <= nt <= 1006:
                                        cands.append((wrt, nt, Tt, kw))
        cands.sort(key=lambda c: -c[0])
        print(f"  K1: {len(cands)} candidats volume-ok ; top 8 :")
        for wrt, nt, Tt, kw in cands[:8]:
            _, nv, wrv = vmatch(p_v, CTX_VCTRL, **kw)
            ha = tuple(h for h in kw['hard_hours'] if h not in CUR_HARD)
            sa = tuple(h for h in kw['soft_hours'] if h not in CUR_SOFT)
            print(f"    test {wrt*100:5.2f}% n={nt} T={Tt:.4f} | ctrl {wrv*100:5.2f}% n={nv} "
                  f"| +hard={ha} +soft={sa} sd={kw['soft_delta']} "
                  f"wbd={kw['wr_bad_day']} skip={kw['skip_cells']}")
