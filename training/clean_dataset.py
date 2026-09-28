from pathlib import Path
from urllib.parse import urlparse
import re

import pandas as pd



BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "dataset" / "phising_urls.csv"
OUTPUT_PATH = BASE_DIR / "dataset" / "cleaned_urls.csv"



def clean_url_value(url):
    """
    Convert Markdown-style URLs such as:

    [https://example.com](https://example.com)

    into:

    https://example.com
    """

    url = str(url).strip()

    match = re.match(
        r"^\[.*?\]\((https?://.*?)\)$",
        url
    )

    if match:
        url = match.group(1)

    return url.strip()



def get_hostname(url):
    """
    Extract hostname from a URL.
    """

    value = str(url).strip()

    if "://" not in value:
        value = "http://" + value

    try:
        hostname = urlparse(value).hostname
    except ValueError:
        return ""

    if hostname is None:
        return ""

    return hostname.lower().rstrip(".")



print("=" * 60)
print("LOADING RAW DATASET")
print("=" * 60)

print("\nInput file:")
print(INPUT_PATH)


if not INPUT_PATH.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_PATH}\n\n"
        "Make sure phising_urls.csv is inside the dataset folder."
    )


df = pd.read_csv(INPUT_PATH)


print("\nRaw dataset shape:")
print(df.shape)


print("\nRaw columns:")
print(df.columns.tolist())



required_columns = {
    "URL",
    "label"
}

missing_columns = (
    required_columns
    - set(df.columns)
)


if missing_columns:

    raise ValueError(
        f"\nMissing required columns: "
        f"{missing_columns}\n\n"
        f"Available columns: "
        f"{df.columns.tolist()}"
    )



print("\nRaw label values:")

print(
    df["label"]
    .value_counts(
        dropna=False
    )
)



df = df[
    ["URL", "label"]
].copy()



df["URL"] = (
    df["URL"]
    .map(clean_url_value)
)



df = df[
    df["URL"].ne("")
].copy()



df = df[
    ~df["URL"]
    .str.lower()
    .isin(
        [
            "nan",
            "none",
            "null"
        ]
    )
].copy()



labels = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)



label_mapping = {

    "benign": 0,
    "legitimate": 0,
    "legit": 0,
    "safe": 0,

    "phishing": 1,
    "malicious": 1,
    "malware": 1,
}


df["label"] = labels.map(
    label_mapping
)



unknown_mask = df["label"].isna()


if unknown_mask.any():

    unknown_values = (
        labels[unknown_mask]
        .value_counts()
    )

    print(
        "\nUnknown label values:"
    )

    print(
        unknown_values
    )

    raise ValueError(
        "\nUnknown labels found in dataset. "
        "Update label_mapping before continuing."
    )



df["label"] = (
    df["label"]
    .astype(int)
)



before_duplicates = len(df)


df = (
    df
    .drop_duplicates(
        subset=["URL"]
    )
    .reset_index(
        drop=True
    )
)


after_duplicates = len(df)


print(
    "\nDuplicate URLs removed:",
    before_duplicates - after_duplicates
)



df["Domain"] = (
    df["URL"]
    .map(get_hostname)
)



invalid_domain_mask = (
    df["Domain"].eq("")
)


invalid_domain_count = (
    invalid_domain_mask.sum()
)


if invalid_domain_count > 0:

    print(
        "\nInvalid URLs removed:",
        invalid_domain_count
    )

    df = df[
        ~invalid_domain_mask
    ].copy()



df["length"] = (
    df["URL"].str.len()
)



if len(df) == 0:

    raise ValueError(
        "\nERROR: CLEANED DATASET IS EMPTY.\n"
        "Check your raw phising_urls.csv file."
    )


if df["label"].nunique() < 2:

    raise ValueError(
        "\nERROR: Dataset contains only one class.\n"
        "Both benign and phishing URLs are required."
    )



print(
    "\n" + "=" * 60
)

print(
    "CLEAN DATASET CREATED"
)

print(
    "=" * 60
)


print(
    "\nFinal dataset shape:"
)

print(
    df.shape
)


print(
    "\nLabel distribution:"
)

print(
    df["label"]
    .value_counts()
    .sort_index()
)


print(
    "\nLabel meaning:"
)

print(
    "0 = BENIGN"
)

print(
    "1 = PHISHING"
)



markdown_urls = df[
    df["URL"].str.match(
        r"^\[.*?\]\(https?://.*\)$"
    )
]


print(
    "\nRemaining Markdown-style URLs:"
)

print(
    len(markdown_urls)
)



print(
    "\nFirst 5 cleaned URLs:"
)

for url in df["URL"].head(5):

    print(
        " ",
        url
    )



OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_PATH
)


print(
    "\nCleaning completed successfully."
)