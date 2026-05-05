import xgboost as xgb
import pandas as pd

from config import MODEL_OUTPUT, BREAKDOWN_THRESHOLD

_model = None


def get_model():
    """Charge le modèle XGBoost depuis le disque (lazy, mis en cache)."""
    global _model
    if _model is None:
        m = xgb.XGBClassifier()
        m.load_model(MODEL_OUTPUT)
        _model = m
    return _model


def _to_dataframe(features):
    if isinstance(features, pd.DataFrame):
        return features
    if isinstance(features, dict):
        return pd.DataFrame([features])
    if isinstance(features, list):
        return pd.DataFrame(features)
    raise TypeError(f"features doit être dict | list[dict] | DataFrame, reçu {type(features).__name__}")


def predict_proba(features):
    """
    Probabilité de retournement (classe 1) pour chaque ligne.

    features : dict (1 setup) | list[dict] (batch) | pd.DataFrame.
    Doit contenir au minimum les colonnes utilisées à l'entraînement.
    Les autres colonnes (timestamp, label, ...) sont ignorées.
    """
    model = get_model()
    df = _to_dataframe(features)
    feature_cols = model.get_booster().feature_names
    return model.predict_proba(df[feature_cols])[:, 1]


def should_take_bet(features, threshold=BREAKDOWN_THRESHOLD):
    """Renvoie un booléen (ou un array de bool) : prob >= threshold."""
    return predict_proba(features) >= threshold


if __name__ == "__main__":
    from config import CSV_PATH
    last = pd.read_csv(CSV_PATH).sort_values("timestamp").tail(1)
    p = float(predict_proba(last)[0])
    take = p >= BREAKDOWN_THRESHOLD
    print(f"Dernière bougie : proba retournement = {p:.4f}  | seuil = {BREAKDOWN_THRESHOLD}  | take = {take}")
