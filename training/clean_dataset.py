from pathlib import Path

import pandas as pd

from feature_extraction import clean_input_url


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "dataset" / "phising_urls.csv"
OUTPUT_PATH = BASE_DIR / "dataset" / "processed" / "cleaned_urls.csv"

LABEL_MAPPING = {
    "0": 0,
    "benign": 0,
    "legitimate": 0,
    "legit": 0,
    "safe": 0,
    "false": 0,
    "1": 1,
    "phishing": 1,
    "malicious": 1,
    "malware": 1,
    "true": 1,
}


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    raw = pd.read_csv(RAW_PATH)
    columns = {column.strip().lower(): column for column in raw.columns}
    if "url" not in columns or "label" not in columns:
        raise ValueError("Raw dataset must contain URL and label columns.")

    df = raw[[columns["url"], columns["label"]]].copy()
    df.columns = ["URL", "label"]
    input_count = len(df)
    missing_url_count = int(df["URL"].isna().sum())
    missing_label_count = int(df["label"].isna().sum())
    df = df.dropna(subset=["URL", "label"]).copy()

    def normalize_url(value):
        try:
            return clean_input_url(value)
        except (TypeError, ValueError):
            return None

    df["URL"] = df["URL"].map(normalize_url)
    invalid_url_count = int(df["URL"].isna().sum())
    df = df.dropna(subset=["URL"]).copy()

    normalized_labels = df["label"].astype("string").str.strip().str.lower()
    df["label"] = normalized_labels.map(LABEL_MAPPING)
    invalid_label_count = int(df["label"].isna().sum())
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    label_counts = df.groupby("URL")["label"].nunique()
    conflicting_urls = set(label_counts[label_counts > 1].index)
    conflicting_rows = int(df["URL"].isin(conflicting_urls).sum())
    if conflicting_urls:
        df = df[~df["URL"].isin(conflicting_urls)].copy()

    duplicate_rows = int(df.duplicated(subset=["URL"], keep="first").sum())
    df = df.drop_duplicates(subset=["URL"], keep="first").reset_index(drop=True)

    if df.empty or df["label"].nunique() != 2:
        raise ValueError("Cleaned dataset must contain both benign and phishing URLs.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    benign_count = int((df["label"] == 0).sum())
    phishing_count = int((df["label"] == 1).sum())
    print("DATA CLEANING SUMMARY")
    print(f"Raw rows: {input_count:,}")
    print(f"Clean rows: {len(df):,}")
    print(f"Benign: {benign_count:,}")
    print(f"Phishing: {phishing_count:,}")
    print(f"Missing URLs removed: {missing_url_count:,}")
    print(f"Missing labels removed: {missing_label_count:,}")
    print(f"Invalid URLs removed: {invalid_url_count:,}")
    print(f"Invalid labels removed: {invalid_label_count:,}")
    print(f"Rows removed for conflicting URLs: {conflicting_rows:,}")
    print(f"Duplicate rows removed: {duplicate_rows:,}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()