from datetime import datetime
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "processed"
MODEL_DIR = BASE_DIR / "model"
MODELS_DIR = BASE_DIR / "models"
ANALYSIS_DIR = BASE_DIR / "analysis"
SPLITS = {
    "train": DATA_DIR / "train.csv",
    "validation": DATA_DIR / "validation.csv",
    "test": DATA_DIR / "test.csv",
}
RANDOM_STATE = 42
MODEL_PATH = MODEL_DIR / "phishing_model_final.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def metrics_for(y_true, predictions):
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def estimator_score(estimator, features):
    if hasattr(estimator, "predict_proba"):
        positive_index = list(estimator.classes_).index(1)
        return estimator.predict_proba(features)[:, positive_index]
    positive_index = list(estimator.classes_).index(1)
    return estimator.decision_function(features) * (1 if positive_index == 1 else -1)


def model_definitions():
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Linear SVM": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=5000,
            )),
        ]),
    }


def importance_for(estimator, feature_names):
    classifier = estimator
    if isinstance(estimator, Pipeline):
        classifier = estimator.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = np.abs(classifier.coef_).mean(axis=0)
    else:
        return None

    return pd.DataFrame({
        "feature": feature_names,
        "importance": values,
    }).sort_values("importance", ascending=False)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    frames = {name: pd.read_csv(path) for name, path in SPLITS.items()}
    feature_names = [
        column for column in frames["train"].columns
        if column not in {"url", "domain", "group", "label"}
    ]
    if not feature_names:
        raise RuntimeError("No model features found in split files.")
    for name, frame in frames.items():
        other_features = [
            column for column in frame.columns
            if column not in {"url", "domain", "group", "label"}
        ]
        if other_features != feature_names:
            raise RuntimeError(f"Feature order mismatch in {name} split.")
        if frame[feature_names].isna().any().any():
            raise RuntimeError(f"Missing numeric features in {name} split.")

    X_train = frames["train"][feature_names]
    y_train = frames["train"]["label"].astype(int)
    X_validation = frames["validation"][feature_names]
    y_validation = frames["validation"]["label"].astype(int)
    X_test = frames["test"][feature_names]
    y_test = frames["test"]["label"].astype(int)

    print("DATASET")
    print(f"Total records: {sum(len(frame) for frame in frames.values()):,}")
    print(f"Train: {len(frames['train']):,}; validation: {len(frames['validation']):,}; test: {len(frames['test']):,}")
    print(f"Benign: {(y_train.eq(0).sum() + y_validation.eq(0).sum() + y_test.eq(0).sum()):,}")
    print(f"Phishing: {(y_train.eq(1).sum() + y_validation.eq(1).sum() + y_test.eq(1).sum()):,}")
    print(f"Features: {len(feature_names)}")

    candidates = {}
    validation_metrics = {}
    for name, estimator in model_definitions().items():
        print(f"\nTraining {name}...")
        estimator.fit(X_train, y_train)
        validation_scores = estimator_score(estimator, X_validation)
        validation_predictions = (validation_scores >= 0.5).astype(int) if hasattr(estimator, "predict_proba") else (validation_scores >= 0).astype(int)
        validation_metrics[name] = metrics_for(y_validation, validation_predictions)
        candidates[name] = estimator
        print(validation_metrics[name])

    selected_name = max(
        validation_metrics,
        key=lambda name: (
            validation_metrics[name]["f1"],
            -validation_metrics[name]["fp"],
            -validation_metrics[name]["fn"],
        ),
    )
    selected = candidates[selected_name]

    comparison_rows = []
    report_lines = ["Held-out test evaluation; prediction cutoffs are the model defaults.", ""]
    for name, estimator in candidates.items():
        scores = estimator_score(estimator, X_test)
        predictions = (scores >= 0.5).astype(int) if hasattr(estimator, "predict_proba") else (scores >= 0).astype(int)
        result = metrics_for(y_test, predictions)
        comparison_rows.append({"model": name, **result})
        report_lines.extend([
            name,
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["Benign", "Phishing"],
                zero_division=0,
            ),
            f"Confusion matrix (rows=true [benign, phishing], columns=predicted):\n{confusion_matrix(y_test, predictions, labels=[0, 1])}",
            "",
        ])

    comparison = pd.DataFrame(comparison_rows).sort_values("f1", ascending=False)
    comparison.to_csv(ANALYSIS_DIR / "model_comparison.csv", index=False)
    (ANALYSIS_DIR / "evaluation_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    best_predictions = selected.predict(X_test[feature_names])
    selected_metrics = metrics_for(y_test, best_predictions)
    importance = importance_for(selected, feature_names)
    if importance is not None:
        importance.to_csv(ANALYSIS_DIR / "feature_importance.csv", index=False)

    final_model = model_definitions()[selected_name]
    final_model.fit(
        pd.concat([X_train, X_validation], ignore_index=True),
        pd.concat([y_train, y_validation], ignore_index=True),
    )
    compatibility_model_paths = [
        MODEL_DIR / "phishing_model.pkl",
        MODELS_DIR / "phishing_model_final.pkl",
        MODELS_DIR / "phishing_model.pkl",
    ]
    for artifact_path in compatibility_model_paths:
        joblib.dump(final_model, artifact_path)

    compatibility_feature_paths = [
        MODEL_DIR / "feature_names.pkl",
        MODELS_DIR / "feature_names_final.pkl",
        MODELS_DIR / "feature_names.pkl",
    ]
    for artifact_path in compatibility_feature_paths:
        joblib.dump(feature_names, artifact_path)

    joblib.dump(final_model, MODEL_PATH)

    total_counts = pd.concat([y_train, y_validation, y_test]).value_counts()
    metadata = {
        "model_type": selected_name,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "decision_cutoff": 0.5 if hasattr(final_model, "predict_proba") else 0.0,
        "decision_score": "probability" if hasattr(final_model, "predict_proba") else "signed_margin",
        "training_date": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "dataset": {
            "total": int(len(frames["train"]) + len(frames["validation"]) + len(frames["test"])),
            "benign": int(total_counts.get(0, 0)),
            "phishing": int(total_counts.get(1, 0)),
            "train": len(frames["train"]),
            "validation": len(frames["validation"]),
            "test": len(frames["test"]),
        },
        "selected_on": "validation F1; ties broken by fewer false positives then false negatives",
        "validation_metrics": validation_metrics[selected_name],
        "test_metrics": selected_metrics,
        "test_confusion_matrix": confusion_matrix(y_test, best_predictions, labels=[0, 1]).tolist(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSelected on validation only: {selected_name}")
    print("\nHELD-OUT TEST COMPARISON")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved report: {ANALYSIS_DIR / 'evaluation_report.txt'}")
    if importance is not None:
        print("\nTop feature importance:")
        print(importance.head(15).to_string(index=False))


if __name__ == "__main__":
    main()