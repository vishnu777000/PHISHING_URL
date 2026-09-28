from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


BASE_DIR = Path(__file__).resolve().parent.parent


TRAIN_PATH = (
    BASE_DIR
    / "dataset"
    / "train.csv"
)

TEST_PATH = (
    BASE_DIR
    / "dataset"
    / "test.csv"
)



train_df = pd.read_csv(
    TRAIN_PATH
)

test_df = pd.read_csv(
    TEST_PATH
)



X_train = train_df.drop(
    columns=["label"]
)

y_train = train_df["label"].astype(
    int
)


X_test = test_df.drop(
    columns=["label"]
)

y_test = test_df["label"].astype(
    int
)



model = RandomForestClassifier(

    n_estimators=300,

    max_depth=18,

    min_samples_split=8,

    min_samples_leaf=3,

    random_state=42,

    n_jobs=-1

)



print(
    "Training Random Forest..."
)

model.fit(
    X_train,
    y_train
)



y_pred = model.predict(
    X_test
)



accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)



print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)


print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(

        y_test,

        y_pred,

        target_names=[
            "Benign",
            "Phishing"
        ],

        zero_division=0

    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)