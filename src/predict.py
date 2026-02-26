import json
import logging
import datetime
from pathlib import Path

import joblib
import numpy as np

from rules import rule_based_triage, inconsistency_score

_BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = _BASE / "data" / "model.joblib"
META_PATH = _BASE / "data" / "meta.json"
LOG_PATH = _BASE / "data" / "predictions.jsonl"

logger = logging.getLogger(__name__)

# --- Module-level cache (loaded once) ---
_model = None
_meta = None


def _get_model_and_meta():
    global _model, _meta
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        with open(META_PATH, "r") as f:
            _meta = json.load(f)
    return _model, _meta


# --- Input validation ---
_RANGES = {
    "age": (0, 120),
    "hr": (20, 250),
    "sbp": (40, 300),
    "dbp": (20, 200),
    "resp_rate": (4, 70),
    "temp": (30.0, 45.0),
    "spo2": (50, 100),
    "pain": (0, 10),
    "chest_pain_severity": (0, 3),
    "sob_severity": (0, 3),
    "confusion": (0, 1),
    "comorbidity_cardiac": (0, 1),
    "comorbidity_respiratory": (0, 1),
    "comorbidity_diabetes": (0, 1),
    "num_comorbidities": (0, 3),
}


def _validate_input(v):
    """Validate and clamp input values to plausible ranges."""
    errors = []
    for key, (lo, hi) in _RANGES.items():
        val = v.get(key)
        if val is None:
            continue
        if not isinstance(val, (int, float)):
            errors.append(f"{key}: valeur non numerique ({val})")
            continue
        if val < lo or val > hi:
            errors.append(f"{key}={val} hors limites [{lo}, {hi}]")
            v[key] = max(lo, min(hi, val))
    return errors


def _compute_confidence(proba, inc):
    """
    Margin-based confidence:
    60% margin(top1 - top2) + 40% max_prob, penalized by entropy.
    Returns (confidence_score, label).
    """
    sorted_p = np.sort(proba)[::-1]
    max_prob = sorted_p[0]
    margin = sorted_p[0] - sorted_p[1] if len(sorted_p) > 1 else sorted_p[0]

    # Entropy penalty (normalized)
    entropy = -np.sum(proba * np.log(proba + 1e-10))
    max_entropy = np.log(len(proba))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

    raw = 0.6 * margin + 0.4 * max_prob
    confidence = raw * (1.0 - 0.3 * norm_entropy)

    # Inconsistency penalty
    if inc >= 0.9:
        confidence *= 0.80
    elif inc >= 0.5:
        confidence *= 0.90

    confidence = max(0.0, min(1.0, confidence))

    # Label
    if confidence >= 0.75:
        label = "Elevee"
    elif confidence >= 0.50:
        label = "Moderee"
    else:
        label = "Faible"

    return round(confidence, 3), label


def _top_features(v, meta, n=3):
    """Return top n contributing features based on feature importance."""
    importance = meta.get("feature_importance", {})
    if not importance:
        return []

    contributions = []
    for feat, imp in importance.items():
        val = v.get(feat)
        if val is None:
            continue
        contributions.append({
            "feature": feat,
            "value": val,
            "importance": round(imp, 4)
        })

    contributions.sort(key=lambda x: x["importance"], reverse=True)
    return contributions[:n]


def _log_prediction(v, result):
    """Append prediction to JSONL log file."""
    try:
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "input": {k: v.get(k) for k in _RANGES},
            "result": {
                "source": result["source"],
                "triage_level": result["triage_level"],
                "confidence": result["confidence"],
            }
        }
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log prediction: {e}")


def predict_triage(v):
    """
    Main prediction function.
    v: dict with patient vitals and symptoms.
    Returns dict with triage_level, confidence, reasons, etc.
    """
    try:
        # Ensure num_comorbidities is computed
        if "num_comorbidities" not in v or v["num_comorbidities"] is None:
            v["num_comorbidities"] = (
                v.get("comorbidity_cardiac", 0)
                + v.get("comorbidity_respiratory", 0)
                + v.get("comorbidity_diabetes", 0)
            )

        # Validate input
        validation_errors = _validate_input(v)

        # 1) Safety rules first
        lvl, reasons = rule_based_triage(v)
        inc = float(inconsistency_score(v))

        if lvl is not None:
            result = {
                "source": "rules",
                "triage_level": int(lvl),
                "confidence": 0.95,
                "confidence_label": "Elevee",
                "reasons": reasons,
                "top_features": [],
                "inconsistency": round(inc, 3),
                "validation_warnings": validation_errors,
                "disclaimer": "Aide a la decision. Ne remplace pas le jugement clinique."
            }
            _log_prediction(v, result)
            return result

        # 2) ML prediction
        model, meta = _get_model_and_meta()
        features = meta["features"]

        import pandas as pd
        row_data = {f: [v.get(f, 0)] for f in features}
        row_df = pd.DataFrame(row_data)
        pred = int(model.predict(row_df)[0])

        proba = np.zeros(5)
        if hasattr(model, "predict_proba"):
            raw_proba = model.predict_proba(row_df)[0]
            classes = model.classes_
            for i, cls in enumerate(classes):
                idx = int(cls) - 1
                if 0 <= idx < 5:
                    proba[idx] = raw_proba[i]

        confidence, conf_label = _compute_confidence(proba, inc)

        reasons = ["Prediction par modele ML (aucun seuil critique declenche)."]
        if inc >= 0.9:
            reasons.append("Inconsistance elevee : severite rapportee non concordante avec les vitaux.")
        elif inc >= 0.5:
            reasons.append("Inconsistance moderee detectee.")

        if validation_errors:
            reasons.append(f"Valeurs corrigees : {', '.join(validation_errors)}")

        top_feat = _top_features(v, meta, n=3)

        result = {
            "source": "ml",
            "triage_level": pred,
            "confidence": confidence,
            "confidence_label": conf_label,
            "reasons": reasons,
            "top_features": top_feat,
            "inconsistency": round(inc, 3),
            "validation_warnings": validation_errors,
            "disclaimer": "Aide a la decision. Ne remplace pas le jugement clinique."
        }
        _log_prediction(v, result)
        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return {
            "source": "error",
            "triage_level": 3,
            "confidence": 0.0,
            "confidence_label": "Faible",
            "reasons": [f"Erreur interne : {str(e)}", "Niveau 3 attribue par defaut de securite."],
            "top_features": [],
            "inconsistency": 0.0,
            "validation_warnings": [],
            "disclaimer": "Aide a la decision. Ne remplace pas le jugement clinique."
        }


if __name__ == "__main__":
    # Normal case
    sample = {
        "age": 72, "hr": 112, "sbp": 128, "dbp": 78, "resp_rate": 22,
        "temp": 37.6, "spo2": 95, "pain": 7,
        "chest_pain_severity": 2, "sob_severity": 0, "confusion": 0,
        "comorbidity_cardiac": 1, "comorbidity_respiratory": 0,
        "comorbidity_diabetes": 0
    }
    r = predict_triage(sample)
    print("=== Normal case ===")
    print(json.dumps(r, indent=2, ensure_ascii=False))

    # Red flag case
    sample2 = {
        "age": 40, "hr": 105, "sbp": 110, "dbp": 70, "resp_rate": 34,
        "temp": 37.0, "spo2": 88, "pain": 3,
        "chest_pain_severity": 0, "sob_severity": 2, "confusion": 0,
        "comorbidity_cardiac": 0, "comorbidity_respiratory": 1,
        "comorbidity_diabetes": 0
    }
    r2 = predict_triage(sample2)
    print("\n=== Red flag case ===")
    print(json.dumps(r2, indent=2, ensure_ascii=False))

    # Infant case
    sample3 = {
        "age": 1, "hr": 170, "sbp": 70, "dbp": 40, "resp_rate": 45,
        "temp": 38.8, "spo2": 91, "pain": 0,
        "chest_pain_severity": 0, "sob_severity": 1, "confusion": 0,
        "comorbidity_cardiac": 0, "comorbidity_respiratory": 0,
        "comorbidity_diabetes": 0
    }
    r3 = predict_triage(sample3)
    print("\n=== Infant case ===")
    print(json.dumps(r3, indent=2, ensure_ascii=False))
