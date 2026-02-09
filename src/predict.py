import json
import joblib
from rules import rule_based_triage, inconsistency_score

MODEL_PATH = "data/model.joblib"
META_PATH = "data/meta.json"


def load_model_and_meta():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return model, meta


def predict_triage(v):
    """
    v: dict with keys:
    age, hr, sbp, dbp, resp_rate, temp, spo2, pain, chest_pain, sob, confusion

    Returns:
      {source, triage_level, confidence, reasons, inconsistency, disclaimer}
    """
    # 1) Safety rules first
    lvl, reasons = rule_based_triage(v)
    inc = float(inconsistency_score(v))

    if lvl is not None:
        return {
            "source": "rules",
            "triage_level": int(lvl),
            "confidence": 1.0,
            "reasons": reasons,
            "inconsistency": round(inc, 3),
            "disclaimer": "Educational decision-support only; not medical advice."
        }

    # 2) ML fallback
    model, meta = load_model_and_meta()
    features = meta["features"]

    row = [[v.get(f) for f in features]]
    pred = int(model.predict(row)[0])

    confidence = 0.5
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0]
        confidence = float(proba.max())

    reasons = ["ML prediction (no red flags triggered)."]

    # Apply small confidence penalty if inconsistency is high
    if inc >= 0.9:
        reasons.append("High inconsistency detected: self-reported severity may be unreliable -> lower confidence.")
        confidence = max(0.0, confidence - 0.15)
    elif inc >= 0.5:
        reasons.append("Some inconsistency detected -> slightly lower confidence.")
        confidence = max(0.0, confidence - 0.08)

    return {
        "source": "ml",
        "triage_level": pred,
        "confidence": round(confidence, 3),
        "reasons": reasons,
        "inconsistency": round(inc, 3),
        "disclaimer": "Educational decision-support only; not medical advice."
    }


if __name__ == "__main__":
    # Normal example
    sample = {
        "age": 72, "hr": 112, "sbp": 128, "dbp": 78, "resp_rate": 22,
        "temp": 37.6, "spo2": 95, "pain": 7, "chest_pain": 1, "sob": 0, "confusion": 0
    }
    print(predict_triage(sample))

    # Red flag example
    sample2 = {
        "age": 40, "hr": 105, "sbp": 110, "dbp": 70, "resp_rate": 34,
        "temp": 37.0, "spo2": 88, "pain": 3, "chest_pain": 0, "sob": 1, "confusion": 0
    }
    print(predict_triage(sample2))