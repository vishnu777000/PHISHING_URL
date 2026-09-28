from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "dataset" / "features.csv"

GRAPH_DIR = BASE_DIR / "analysis"
GRAPH_DIR.mkdir(exist_ok=True)



df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("PHISHING URL DATASET - FEATURE ANALYSIS")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)



print("\n" + "=" * 60)
print("COLUMNS")
print("=" * 60)

print(df.columns.tolist())



print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)



print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())



print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)

print("Duplicate rows:", df.duplicated().sum())



print("\n" + "=" * 60)
print("LABEL DISTRIBUTION")
print("=" * 60)

label_counts = df["label"].value_counts().sort_index()

print(label_counts)

print("\nLabel percentages:")
print(
    df["label"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
)



print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe())



print("\n" + "=" * 60)
print("FEATURE COMPARISON")
print("=" * 60)

comparison = df.groupby("label").mean(numeric_only=True)

print(comparison)



print("\n" + "=" * 60)
print("LABEL MEANING")
print("=" * 60)

print("0 = Benign")
print("1 = Phishing")



plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="label"
)

plt.title("Benign vs Phishing URLs")
plt.xlabel("URL Type")
plt.ylabel("Number of URLs")

plt.xticks(
    [0, 1],
    ["Benign", "Phishing"]
)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "label_distribution.png",
    dpi=300
)

plt.show()



plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="label",
    y="url_length"
)

plt.title("URL Length: Benign vs Phishing")
plt.xlabel("URL Type")
plt.ylabel("URL Length")

plt.xticks(
    [0, 1],
    ["Benign", "Phishing"]
)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "url_length_comparison.png",
    dpi=300
)

plt.show()



correlation = df.corr(numeric_only=True)

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "correlation_matrix.png",
    dpi=300
)

plt.show()



print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print("\nGraphs saved in:")
print(GRAPH_DIR)