import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV
)
from sklearn.metrics import (
    classification_report, confusion_matrix, make_scorer,
    recall_score
)
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform

from pathlib import Path
_BASE = Path(__file__).resolve().parent.parent

DATA_PATH = str(_BASE / "data" / "triage.csv")
MODEL_PATH = str(_BASE / "data" / "model.joblib")
META_PATH = str(_BASE / "data" / "meta.json")

FEATURES = [
    "age", "hr", "sbp", "dbp", "resp_rate", "temp", "spo2",
    "pain", "chest_pain_severity", "sob_severity", "confusion",
    "comorbidity_cardiac", "comorbidity_respiratory",
    "comorbidity_diabetes", "num_comorbidities"
]
TARGET = "triage_level"


def _critical_recall_scorer(y_true, y_pred):
    """Custom scorer that prioritizes recall on levels 1-2."""
    # Macro recall across all classes
    macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    # Recall on critical levels (1 and 2)
    mask_1 = y_true == 1
    mask_2 = y_true == 2
    rec_1 = (y_pred[mask_1] == 1).mean() if mask_1.sum() > 0 else 0
    rec_2 = (y_pred[mask_2] == 2).mean() if mask_2.sum() > 0 else 0
    # Weighted: 40% critical recall + 60% macro
    return 0.4 * (0.6 * rec_1 + 0.4 * rec_2) + 0.6 * macro


critical_scorer = make_scorer(_critical_recall_scorer)


def main():
    t0 = time.time()
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    print(f"Dataset: {len(df)} samples, {len(FEATURES)} features")
    print(f"Class distribution:\n{y.value_counts().sort_index()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Base estimators ---
    gb = GradientBoostingClassifier(random_state=42)
    rf = RandomForestClassifier(
        class_weight='balanced', random_state=42, n_jobs=-1
    )

    # VotingClassifier (soft voting for probability output)
    voting = VotingClassifier(
        estimators=[('gb', gb), ('rf', rf)],
        voting='soft'
    )

    # Pipeline with scaling
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', voting)
    ])

    # --- Hyperparameter search space ---
    param_dist = {
        'clf__gb__n_estimators': randint(100, 300),
        'clf__gb__max_depth': randint(4, 8),
        'clf__gb__learning_rate': uniform(0.05, 0.15),
        'clf__gb__min_samples_split': randint(5, 20),
        'clf__gb__subsample': uniform(0.7, 0.3),
        'clf__rf__n_estimators': randint(100, 300),
        'clf__rf__max_depth': randint(8, 20),
        'clf__rf__min_samples_split': randint(5, 15),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Running RandomizedSearchCV (20 iterations, 5-fold)...")
    search = RandomizedSearchCV(
        pipe, param_dist,
        n_iter=20,
        cv=cv,
        scoring=critical_scorer,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    search.fit(X_train, y_train)

    best_pipe = search.best_estimator_
    print(f"\nBest CV score: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    # --- Isotonic calibration ---
    print("\nCalibrating probabilities (isotonic)...")
    calibrated = CalibratedClassifierCV(
        best_pipe, method='isotonic', cv=5
    )
    calibrated.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = calibrated.predict(X_test)
    y_proba = calibrated.predict_proba(X_test)

    report_str = classification_report(y_test, y_pred, digits=3)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Classification report ===")
    print(report_str)
    print("\n=== Confusion matrix (rows=true, cols=pred) ===")
    print(cm)

    # Critical recall
    rec_1 = report_dict.get('1', {}).get('recall', 0)
    rec_2 = report_dict.get('2', {}).get('recall', 0)
    print(f"\nRecall L1: {rec_1:.3f}")
    print(f"Recall L2: {rec_2:.3f}")
    print(f"Recall L1+L2 avg: {(rec_1 + rec_2) / 2:.3f}")

    # --- Feature importance (from best uncalibrated model) ---
    voting_clf = best_pipe.named_steps['clf']
    gb_model = voting_clf.named_estimators_['gb']
    rf_model = voting_clf.named_estimators_['rf']

    gb_imp = gb_model.feature_importances_
    rf_imp = rf_model.feature_importances_
    avg_imp = (gb_imp + rf_imp) / 2.0

    feat_importance = {
        f: round(float(avg_imp[i]), 4) for i, f in enumerate(FEATURES)
    }
    # Sort by importance
    feat_importance = dict(
        sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)
    )

    print("\n=== Feature importance (avg GB+RF) ===")
    for f, imp in feat_importance.items():
        print(f"  {f:30s} {imp:.4f}")

    # --- Save model ---
    joblib.dump(calibrated, MODEL_PATH)

    elapsed = time.time() - t0

    meta = {
        "features": FEATURES,
        "target": TARGET,
        "model": "CalibratedClassifierCV(VotingClassifier(GradientBoosting + RandomForest))",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "best_cv_score": round(search.best_score_, 4),
        "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else
                           float(v) if isinstance(v, (np.floating,)) else v)
                        for k, v in search.best_params_.items()},
        "classification_report": {
            k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                for kk, vv in v.items()} if isinstance(v, dict) else round(v, 4)
            for k, v in report_dict.items()
        },
        "recall_L1": round(rec_1, 4),
        "recall_L2": round(rec_2, 4),
        "feature_importance": feat_importance,
        "training_time_seconds": round(elapsed, 1),
        "note": "Educational triage decision-support. Not medical advice."
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved meta  -> {META_PATH}")
    print(f"Training completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
