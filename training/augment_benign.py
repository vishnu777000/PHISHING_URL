from pathlib import Path

import pandas as pd

from feature_extraction import clean_input_url


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "dataset" / "processed" / "cleaned_urls.csv"
OUTPUT_PATH = BASE_DIR / "dataset" / "processed" / "augmented_urls.csv"
RANDOM_SEED = 42

UNIVERSITY_DOMAINS = (
    "annauniv.edu",
    "osmania.ac.in",
    "jnu.ac.in",
    "iitd.ac.in",
    "iitm.ac.in",
    "iisc.ac.in",
    "du.ac.in",
    "bits-pilani.ac.in",
    "vit.ac.in",
    "nitt.edu",
)

UNIVERSITY_PATHS = (
    "",
    "/academics",
    "/students",
    "/examinations",
    "/results",
    "/departments/computer-science",
    "/faculty",
    "/research",
    "/contact",
    "/admissions/undergraduate",
    "/student-portal/login",
    "/examinations/results/2025",
)

OTHER_BENIGN_URLS = (
    "https://accounts.google.com/ServiceLogin",
    "https://github.com/settings/profile",
    "https://www.google.com/search?q=url+classification",
    "https://www.youtube.com/results?search_query=computer+science",
    "https://www.amazon.in/s?k=notebooks&sort=price-asc",
    "https://docs.python.org/3/library/urllib.parse.html",
    "https://mail.google.com/mail/u/0/",
    "https://login.microsoftonline.com/common/oauth2/",
    "https://www.amazon.in/ap/account",
    "https://www.wikipedia.org/wiki/Computer_science",
)


def generate_benign_variations():
    urls = set(OTHER_BENIGN_URLS)
    for domain in UNIVERSITY_DOMAINS:
        for host in (domain, f"www.{domain}"):
            for path in UNIVERSITY_PATHS:
                urls.add(f"https://{host}{path}")
    return sorted(urls)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Clean dataset not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    if not {"URL", "label"}.issubset(df.columns):
        raise ValueError("Clean dataset must contain URL and label columns.")

    df["URL"] = df["URL"].map(clean_input_url)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    existing_urls = set(df["URL"])

    candidates = pd.DataFrame({
        "URL": generate_benign_variations(),
        "label": 0,
    })
    candidates["URL"] = candidates["URL"].map(clean_input_url)
    candidates = candidates[~candidates["URL"].isin(existing_urls)]

    combined = pd.concat([df[["URL", "label"]], candidates], ignore_index=True)
    if not combined["URL"].is_unique:
        raise RuntimeError("Duplicate URLs remain after augmentation.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    benign = int((combined["label"] == 0).sum())
    phishing = int((combined["label"] == 1).sum())
    print("BENIGN AUGMENTATION SUMMARY")
    print(f"Source rows: {len(df):,}")
    print(f"New benign URLs: {len(candidates):,}")
    print(f"Final rows: {len(combined):,}")
    print(f"Benign: {benign:,} ({benign / len(combined):.2%})")
    print(f"Phishing: {phishing:,} ({phishing / len(combined):.2%})")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()