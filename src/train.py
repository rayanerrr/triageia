import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/triage.csv"
MODEL_PATH = "data/model.joblib"
META_PATH = "data/meta.json"

FEATURES = [
    "age", "hr", "sbp", "dbp", "resp_rate", "temp", "spo2",
    "pain", "chest_pain", "sob", "confusion"
]
TARGET = "triage_level"

def main():
    df = pd.read_csv(DATA_PATH)

    # Ensure required columns exist
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Interpretable baseline model
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000))
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, digits=3))

    print("\n=== Confusion matrix (rows=true, cols=pred) ===")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(pipe, MODEL_PATH)

    meta = {
        "features": FEATURES,
        "target": TARGET,
        "model": "StandardScaler + LogisticRegression",
        "note": "Educational triage decision-support. Not medical advice."
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved meta  -> {META_PATH}")

if __name__ == "__main__":
    main()