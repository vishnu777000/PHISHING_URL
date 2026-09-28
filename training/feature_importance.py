from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "dataset" / "features.csv"

GRAPH_DIR = BASE_DIR / "analysis"

GRAPH_DIR.mkdir(
    exist_ok=True
)



df = pd.read_csv(INPUT_PATH)

X = df.drop(
    columns=["label"]
)

y = df["label"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)



model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)



importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})


importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)



print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance_df.to_string(
        index=False
    )
)



importance_df.to_csv(
    BASE_DIR / "feature_importance.csv",
    index=False
)



top_features = importance_df.head(15)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 15 Features for Phishing URL Detection"
)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "feature_importance.png",
    dpi=300
)

plt.show()