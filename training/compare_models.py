from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)



BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "dataset" / "features.csv"



df = pd.read_csv(INPUT_PATH)

X = df.drop(columns=["label"])
y = df["label"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)



scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_pred = logistic_model.predict(
    X_test_scaled
)



tree_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5
)

tree_model.fit(
    X_train,
    y_train
)

tree_pred = tree_model.predict(
    X_test
)



forest_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

forest_model.fit(
    X_train,
    y_train
)

forest_pred = forest_model.predict(
    X_test
)



def calculate_metrics(y_true, prediction):

    return {
        "Accuracy": accuracy_score(
            y_true,
            prediction
        ),

        "Precision": precision_score(
            y_true,
            prediction
        ),

        "Recall": recall_score(
            y_true,
            prediction
        ),

        "F1 Score": f1_score(
            y_true,
            prediction
        )
    }



results = {

    "Logistic Regression":
        calculate_metrics(
            y_test,
            logistic_pred
        ),

    "Decision Tree":
        calculate_metrics(
            y_test,
            tree_pred
        ),

    "Random Forest":
        calculate_metrics(
            y_test,
            forest_pred
        )
}



results_df = pd.DataFrame(results).T

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)



RESULT_PATH = BASE_DIR / "model_comparison.csv"

results_df.to_csv(
    RESULT_PATH
)

print("\nComparison saved to:")
print(RESULT_PATH)