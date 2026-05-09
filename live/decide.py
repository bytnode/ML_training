"""Qualifies setups, calls predict, picks the highest-confidence taken setup.

This is the single place where the live daemon's decision logic lives.
Mirrors the chain used in backtest/modele_15m: predict_proba -> should_take_bet,
then direction = 'down' if seq_dir==1 else 'up' (same mapping as api.py).
"""
import settings  # triggers sys.path injection

# Helpers from backtest/modele_15m/predict.py
from predict import predict_proba, should_take_bet  # noqa: E402


def evaluate_candidates(candidates_with_features: list[dict]) -> list[dict]:
    """Add confidence + taken + bet_direction to each candidate.

    candidates_with_features: list of dicts each having {seq_len, seq_dir (int), features}.
        features is the 42-feature dict from features.build_features.

    Returns list of dicts each with the original fields plus:
      confidence (float), taken (bool), bet_direction ('up'|'down'),
      effective_threshold (float), bucket_v (int), bucket_w (int).
    """
    if not candidates_with_features:
        return []

    feature_dicts = [c["features"] for c in candidates_with_features]
    confidences = predict_proba(feature_dicts)
    takes = should_take_bet(feature_dicts)

    enriched = []
    for cand, conf, take in zip(candidates_with_features, confidences, takes):
        feat = cand["features"]
        # Effective threshold + bucket info, recomputed here for audit (not strictly needed
        # since should_take_bet already applied it, but we want to log it).
        bucket_v = int(feat["volatility_state"])
        bucket_w = int(feat["is_weekend"])
        # Direction mapping: seq_dir=1 (up streak) -> bet reversal = DOWN.
        bet_direction = "down" if int(feat["seq_dir"]) == 1 else "up"

        enriched.append({
            **cand,
            "confidence": float(conf),
            "taken": bool(take),
            "bet_direction": bet_direction,
            "bucket_v": bucket_v,
            "bucket_w": bucket_w,
        })
    return enriched


def pick_chosen(evaluated: list[dict]) -> dict | None:
    """Return the highest-confidence taken candidate, or None if none are taken."""
    eligible = [c for c in evaluated if c["taken"]]
    if not eligible:
        return None
    return max(eligible, key=lambda c: c["confidence"])
