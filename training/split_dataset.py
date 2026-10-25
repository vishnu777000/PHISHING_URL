from pathlib import Path

import pandas as pd
import tldextract
from sklearn.model_selection import StratifiedGroupKFold

from feature_extraction import extract_features


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "processed"
INPUT_PATH = DATA_DIR / "features.csv"
RANDOM_STATE = 42
DOMAIN_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())


def registrable_domain(hostname):
    result = DOMAIN_EXTRACTOR(str(hostname).lower().strip())
    return result.top_domain_under_public_suffix or str(hostname).lower().strip()


def summarize(name, frame):
    counts = frame["label"].value_counts()
    print(
        f"{name:<12} total={len(frame):,} "
        f"benign={int(counts.get(0, 0)):,} "
        f"phishing={int(counts.get(1, 0)):,}"
    )


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Feature dataset not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    required = {"url", "domain", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Feature data is missing columns: {sorted(required - set(df.columns))}")

    features = list(extract_features("example.com"))
    if [column for column in df.columns if column not in {"url", "domain", "label"}] != features:
        raise RuntimeError("Feature names or ordering differ from the inference extractor.")
    if df["url"].duplicated().any():
        raise RuntimeError("Duplicate URLs exist before splitting; rerun cleaning and augmentation.")

    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if set(df["label"].unique()) != {0, 1}:
        raise ValueError("Both benign (0) and phishing (1) labels are required.")

    df["group"] = df["domain"].map(registrable_domain)
    if df["group"].eq("").any():
        raise RuntimeError("Rows with empty registered domains cannot be split safely.")

    first_split = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    train_val_indices, test_indices = next(
        first_split.split(df, df["label"], groups=df["group"])
    )
    train_val = df.iloc[train_val_indices].copy()
    test = df.iloc[test_indices].copy()

    second_split = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    train_indices, validation_indices = next(
        second_split.split(train_val, train_val["label"], groups=train_val["group"])
    )
    train = train_val.iloc[train_indices].copy()
    validation = train_val.iloc[validation_indices].copy()

    frames = {"train": train, "validation": validation, "test": test}
    for split_name, frame in frames.items():
        if frame.empty or frame["label"].nunique() != 2:
            raise RuntimeError(f"{split_name} split must contain both classes.")

    split_names = list(frames)
    for index, first_name in enumerate(split_names):
        first = frames[first_name]
        for second_name in split_names[index + 1:]:
            second = frames[second_name]
            if set(first["url"]) & set(second["url"]):
                raise RuntimeError(f"Exact URL leakage between {first_name} and {second_name}.")
            if set(first["group"]) & set(second["group"]):
                raise RuntimeError(f"Registered-domain leakage between {first_name} and {second_name}.")

    output_columns = ["url", "domain", "group", *features, "label"]
    for split_name, frame in frames.items():
        frame[output_columns].to_csv(DATA_DIR / f"{split_name}.csv", index=False)

    print(f"Total: {len(df):,} | Benign: {(df.label == 0).sum():,} | Phishing: {(df.label == 1).sum():,}")
    for split_name, frame in frames.items():
        summarize(split_name, frame)
        print(f"             registered domains={frame['group'].nunique():,}")
    print("Domain leakage: 0 across all splits")
    print(f"Saved splits under: {DATA_DIR}")


if __name__ == "__main__":
    main()