"""
01_features_diff.py - Diff exhaustif des deux parquets UP/DOWN.

Sortie : analyse/01_features_diff.md
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "01_features_diff.md"

DOWN_PARQUET = Path("../db/btc_multi_tf_15m.parquet")
UP_PARQUET = Path("../../modele_up/db/btc_multi_tf_15m.parquet")


def main():
    df_d = pd.read_parquet(DOWN_PARQUET)
    df_u = pd.read_parquet(UP_PARQUET)

    cols_d = set(df_d.columns)
    cols_u = set(df_u.columns)
    common = cols_d & cols_u
    down_only = cols_d - cols_u
    up_only = cols_u - cols_d

    lines = ["# Diff features UP vs DOWN parquet\n\n"]
    lines.append(f"## Shapes\n\n")
    lines.append(f"- DOWN : {len(df_d):,} rows, {len(df_d.columns)} cols\n")
    lines.append(f"- UP   : {len(df_u):,} rows, {len(df_u.columns)} cols\n\n")

    lines.append(f"## Common features ({len(common)})\n\n")
    lines.append(f"```\n")
    for c in sorted(common):
        lines.append(f"  {c}\n")
    lines.append(f"```\n\n")

    lines.append(f"## DOWN-only ({len(down_only)})\n\n")
    lines.append(f"```\n")
    for c in sorted(down_only):
        lines.append(f"  {c}\n")
    lines.append(f"```\n\n")

    lines.append(f"## UP-only ({len(up_only)})\n\n")
    lines.append(f"```\n")
    for c in sorted(up_only):
        lines.append(f"  {c}\n")
    lines.append(f"```\n\n")

    # Check label distribution
    lines.append(f"## Label distribution\n\n")
    for nm, df in [("DOWN", df_d), ("UP", df_u)]:
        label_count = df["label"].value_counts(dropna=False).to_dict()
        n_nan = int(df["label"].isna().sum())
        lines.append(f"### {nm}\n\n")
        lines.append(f"- label=1 : {int(df['label'].sum())}  ({df['label'].mean():.4f})\n")
        lines.append(f"- label=0 : {int((df['label']==0).sum())}  ({(df['label']==0).mean():.4f})\n")
        lines.append(f"- label NaN : {n_nan}\n\n")

    # Check if DOWN's label is exactly (1 - UP's label)
    df_merge = df_d.merge(df_u[["timestamp_ms", "label"]],
                          on="timestamp_ms", suffixes=("_down", "_up"))
    df_merge = df_merge.dropna(subset=["label_down", "label_up"])
    sum_check = (df_merge["label_down"] + df_merge["label_up"])
    n_inverse = int((sum_check == 1).sum())
    n_same = int((sum_check == 0).sum() + (sum_check == 2).sum())
    n_total = len(df_merge)
    lines.append(f"## Label coherence check (on common timestamps)\n\n")
    lines.append(f"- Common timestamps with both labels valid : {n_total:,}\n")
    lines.append(f"- label_down + label_up == 1 (inverse, attendu): {n_inverse:,} ({100*n_inverse/n_total:.2f}%)\n")
    lines.append(f"- label_down + label_up == 0 ou 2 (incoherent): {n_same:,} ({100*n_same/n_total:.2f}%)\n\n")

    # Check common features have same values
    lines.append(f"## Common features value coherence (échantillon 1000 rows)\n\n")
    lines.append(f"| feature | DOWN std | UP std | DOWN mean | UP mean | match? |\n")
    lines.append(f"|---|---|---|---|---|---|\n")
    sample_d = df_d.sample(n=min(1000, len(df_d)), random_state=42)
    sample_u_at_same_ts = df_u[df_u["timestamp_ms"].isin(sample_d["timestamp_ms"])]
    aligned = sample_d.merge(sample_u_at_same_ts, on="timestamp_ms", suffixes=("_d", "_u"))
    for col in sorted(common):
        if col in ("timestamp_ms", "timestamp", "label"):
            continue
        col_d = col + "_d"
        col_u = col + "_u"
        if col_d in aligned.columns and col_u in aligned.columns:
            a = aligned[col_d].dropna()
            b = aligned[col_u].dropna()
            if len(a) == 0 or len(b) == 0:
                continue
            # Try to align indices
            both = aligned.dropna(subset=[col_d, col_u])
            if len(both) == 0:
                continue
            equal = np.allclose(both[col_d].astype(float), both[col_u].astype(float), atol=1e-6)
            lines.append(f"| {col} | {a.std():.4f} | {b.std():.4f} | {a.mean():.4f} | {b.mean():.4f} | {'YES' if equal else 'NO'} |\n")

    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"[ok] -> {OUT}")
    print(f"\nCommon features: {len(common)}")
    print(f"DOWN-only: {len(down_only)} : {sorted(down_only)}")
    print(f"UP-only: {len(up_only)} : {sorted(up_only)}")


if __name__ == "__main__":
    main()
