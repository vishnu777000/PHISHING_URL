from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from collections import Counter

import ipaddress
import math
import re

import pandas as pd



BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "dataset" / "augmented_urls.csv"
OUTPUT_PATH = BASE_DIR / "dataset" / "features.csv"



SUSPICIOUS_WORDS = {
    "login",
    "signin",
    "sign-in",
    "verify",
    "verification",
    "account",
    "update",
    "secure",
    "security",
    "confirm",
    "confirmation",
    "password",
    "credential",
    "credentials",
    "bank",
    "banking",
    "payment",
    "payments",
    "wallet",
    "paypal",
    "authorize",
    "authorization",
    "authentication",
    "recover",
    "recovery",
    "unlock",
    "suspend",
    "suspended",
    "billing",
    "invoice",
    "transaction",
    "support",
    "webscr",
}



SHORTENING_SERVICES = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "is.gd",
    "buff.ly",
    "ow.ly",
    "cutt.ly",
    "shorturl.at",
    "rebrand.ly",
    "tiny.cc",
    "rb.gy",
    "lnkd.in",
}



TRUSTED_TLDS = {
    ".edu",
    ".ac.in",
    ".edu.in",
    ".ac.uk",
    ".edu.au",
    ".ac.jp",
    ".edu.cn",
    ".edu.br",
    ".edu.mx",
    ".ac.nz",
    ".edu.sg",
    ".ac.kr",
    ".edu.pk",
    ".edu.ng",
    ".edu.za",
    ".ac.id",
    ".edu.my",
    ".edu.ph",
    ".edu.tw",
    ".edu.hk",
    ".edu.eg",
    ".edu.tr",
    ".edu.ar",
    ".edu.co",
    ".edu.pe",
    ".ac.th",
    ".gov",
    ".gov.in",
    ".gov.uk",
    ".gov.au",
    ".gov.br",
    ".go.jp",
    ".gob.mx",
    ".gc.ca",
    ".mil",
}



SUSPICIOUS_TLDS = {
    ".tk",
    ".ml",
    ".ga",
    ".cf",
    ".gq",
    ".xyz",
    ".top",
    ".buzz",
    ".club",
    ".work",
    ".click",
    ".link",
    ".info",
    ".icu",
    ".rest",
    ".cam",
}



LOGIN_WORDS = {
    "login",
    "signin",
    "sign-in",
    "log-in",
}

VERIFY_WORDS = {
    "verify",
    "verification",
    "confirm",
    "confirmation",
    "validate",
}

ACCOUNT_WORDS = {
    "account",
    "credential",
    "credentials",
    "password",
}

SECURITY_WORDS = {
    "secure",
    "security",
    "authentication",
    "authorize",
    "authorization",
}

PAYMENT_WORDS = {
    "payment",
    "payments",
    "billing",
    "invoice",
    "transaction",
    "wallet",
}

BANK_WORDS = {
    "bank",
    "banking",
}



BRAND_DOMAINS = {
    'paypal': 'paypal.com',
    'apple': 'apple.com',
    'microsoft': 'microsoft.com',
    'google': 'google.com',
    'facebook': 'facebook.com',
    'amazon': 'amazon.com',
    'netflix': 'netflix.com',
    'instagram': 'instagram.com',
    'whatsapp': 'whatsapp.com',
    'telegram': 'telegram.com',
    'linkedin': 'linkedin.com',
    'twitter': 'twitter.com',
    'chase': 'chase.com',
    'wellsfargo': 'wellsfargo.com',
    'bankofamerica': 'bankofamerica.com',
    'citibank': 'citibank.com',
    'americanexpress': 'americanexpress.com'
}



def clean_input_url(url: str) -> str:

    url = str(url).strip()

    match = re.match(
        r"^\[.*?\]\((https?://.*?)\)$",
        url
    )

    if match:
        url = match.group(1)

    url = url.strip("\"'")

    return url.strip()



def parse_url(url: str):

    url = clean_input_url(url)

    if "://" not in url:
        url = "http://" + url

    parsed = urlsplit(url)

    hostname = parsed.hostname or ""

    hostname = hostname.lower().rstrip(".")

    return parsed, hostname



def calculate_entropy(text: str) -> float:

    if not text:
        return 0.0

    counter = Counter(text)

    length = len(text)

    entropy = 0.0

    for count in counter.values():

        probability = count / length

        entropy -= (
            probability
            * math.log2(probability)
        )

    return entropy



def digit_ratio(text: str) -> float:

    if not text:
        return 0.0

    return sum(
        character.isdigit()
        for character in text
    ) / len(text)


def letter_ratio(text: str) -> float:

    if not text:
        return 0.0

    return sum(
        character.isalpha()
        for character in text
    ) / len(text)


def special_character_ratio(text: str) -> float:

    if not text:
        return 0.0

    return sum(
        not character.isalnum()
        for character in text
    ) / len(text)



def has_trusted_tld(hostname: str) -> int:
    """Check if hostname ends with a trusted/restricted TLD."""

    if not hostname:
        return 0

    hostname = hostname.lower()

    for tld in TRUSTED_TLDS:
        if hostname.endswith(tld):
            return 1

    return 0


def has_suspicious_tld(hostname: str) -> int:
    """Check if hostname ends with a suspicious/free TLD."""

    if not hostname:
        return 0

    hostname = hostname.lower()

    for tld in SUSPICIOUS_TLDS:
        if hostname.endswith(tld):
            return 1

    return 0



def is_ip_address(hostname: str) -> int:

    if not hostname:
        return 0

    try:

        ipaddress.ip_address(hostname)

        return 1

    except ValueError:

        return 0



def contains_any_word(text: str, words: set) -> int:

    text = text.lower()

    for word in words:

        pattern = (
            rf"(?<![a-z])"
            rf"{re.escape(word)}"
            rf"(?![a-z])"
        )

        if re.search(pattern, text):
            return 1

    return 0


def suspicious_word_count(url: str) -> int:

    url_lower = url.lower()

    count = 0

    for word in SUSPICIOUS_WORDS:

        pattern = (
            rf"(?<![a-z])"
            rf"{re.escape(word)}"
            rf"(?![a-z])"
        )

        if re.search(pattern, url_lower):
            count += 1

    return count



def has_repeated_characters(text: str) -> int:

    if not text:
        return 0

    return int(
        bool(re.search(r"(.)\1\1", text))
    )



def extract_features(url: str) -> dict:

    cleaned_url = clean_input_url(url)

    parsed, hostname = parse_url(cleaned_url)


    feature_hostname = hostname

    if feature_hostname.startswith("www."):

        feature_hostname = feature_hostname[4:]


    path = parsed.path or ""

    if path == "/" and "://" in cleaned_url:
        parsed = parsed._replace(path="")
        cleaned_url = urlunsplit(parsed)
        path = ""

    query = parsed.query or ""

    fragment = parsed.fragment or ""


    url_length = len(cleaned_url)

    hostname_length = len(hostname)

    domain_length = len(feature_hostname)

    path_length = len(path)

    query_length = len(query)

    fragment_length = len(fragment)


    num_dots = cleaned_url.count(".")

    num_hyphens = cleaned_url.count("-")

    num_underscores = cleaned_url.count("_")

    num_slashes = cleaned_url.count("/")

    num_backslashes = cleaned_url.count("\\")

    num_digits = sum(
        character.isdigit()
        for character in cleaned_url
    )

    num_letters = sum(
        character.isalpha()
        for character in cleaned_url
    )

    num_special_chars = sum(
        not character.isalnum()
        for character in cleaned_url
    )


    num_question_marks = cleaned_url.count("?")

    num_equals = cleaned_url.count("=")

    num_ampersands = cleaned_url.count("&")

    num_percent = cleaned_url.count("%")

    num_colons = cleaned_url.count(":")

    num_semicolons = cleaned_url.count(";")

    num_at = cleaned_url.count("@")


    domain_parts = [
        part
        for part in feature_hostname.split(".")
        if part
    ]

    has_ip = is_ip_address(hostname)


    if has_ip:

        num_subdomains = 0

    else:

        num_subdomains = max(
            0,
            len(domain_parts) - 2
        )


    domain_digits = sum(
        character.isdigit()
        for character in feature_hostname
    )


    path_segments = [
        segment
        for segment in path.split("/")
        if segment
    ]

    path_depth = len(path_segments)


    has_https = int(
        parsed.scheme.lower() == "https"
    )


    try:

        has_port = int(
            parsed.port is not None
        )

    except ValueError:

        has_port = 1


    has_punycode = int(
        "xn--" in feature_hostname
    )


    is_shortened = int(
        feature_hostname in SHORTENING_SERVICES
    )


    is_trusted_tld = has_trusted_tld(
        feature_hostname
    )

    is_suspicious_tld = has_suspicious_tld(
        feature_hostname
    )


    has_at = int("@" in cleaned_url)

    has_question_mark = int(
        "?" in cleaned_url
    )

    has_fragment = int(
        "#" in cleaned_url
    )

    has_double_slash = int(
        "//" in cleaned_url[
            cleaned_url.find("://") + 3:
        ]
        if "://" in cleaned_url
        else "//" in cleaned_url
    )


    suspicious_words = suspicious_word_count(
        cleaned_url
    )

    has_login_word = contains_any_word(
        cleaned_url,
        LOGIN_WORDS
    )

    has_verify_word = contains_any_word(
        cleaned_url,
        VERIFY_WORDS
    )

    has_account_word = contains_any_word(
        cleaned_url,
        ACCOUNT_WORDS
    )

    has_security_word = contains_any_word(
        cleaned_url,
        SECURITY_WORDS
    )

    has_payment_word = contains_any_word(
        cleaned_url,
        PAYMENT_WORDS
    )

    has_bank_word = contains_any_word(
        cleaned_url,
        BANK_WORDS
    )


    url_entropy = calculate_entropy(
        cleaned_url
    )

    domain_entropy = calculate_entropy(
        feature_hostname
    )

    url_digit_ratio = digit_ratio(
        cleaned_url
    )

    domain_digit_ratio = digit_ratio(
        feature_hostname
    )

    letter_ratio_value = letter_ratio(
        cleaned_url
    )

    special_ratio = special_character_ratio(
        cleaned_url
    )


    repeated_characters = has_repeated_characters(
        cleaned_url
    )


    suspicious_word_in_domain = contains_any_word(
        feature_hostname,
        SUSPICIOUS_WORDS
    )

    suspicious_word_in_path = contains_any_word(
        path,
        SUSPICIOUS_WORDS
    )

    trusted_tld_with_path_keywords = int(
        is_trusted_tld == 1 and suspicious_word_in_path == 1
    )

    domain_has_brand_name = 0
    feature_hostname_lower = feature_hostname.lower()
    for brand, official_domain in BRAND_DOMAINS.items():
        if brand in feature_hostname_lower and not feature_hostname_lower.endswith(official_domain):
            domain_has_brand_name = 1
            break

    path_to_domain_length_ratio = path_length / max(domain_length, 1)


    features = {

        "url_length": url_length,
        "hostname_length": hostname_length,
        "domain_length": domain_length,
        "path_length": path_length,
        "query_length": query_length,
        "fragment_length": fragment_length,
        "path_depth": path_depth,

        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_underscores": num_underscores,
        "num_slashes": num_slashes,
        "num_backslashes": num_backslashes,
        "num_digits": num_digits,
        "num_letters": num_letters,
        "num_special_chars": num_special_chars,

        "num_question_marks": num_question_marks,
        "num_equals": num_equals,
        "num_ampersands": num_ampersands,
        "num_percent": num_percent,
        "num_colons": num_colons,
        "num_semicolons": num_semicolons,
        "num_at": num_at,

        "num_subdomains": num_subdomains,
        "domain_digits": domain_digits,
        "domain_digit_ratio": domain_digit_ratio,

        "has_https": has_https,
        "has_ip": has_ip,
        "has_at": has_at,
        "has_port": has_port,
        "has_punycode": has_punycode,
        "is_shortened": is_shortened,
        "is_trusted_tld": is_trusted_tld,
        "is_suspicious_tld": is_suspicious_tld,
        "has_question_mark": has_question_mark,
        "has_fragment": has_fragment,
        "has_double_slash": has_double_slash,

        "suspicious_word_count": suspicious_words,
        "has_login_word": has_login_word,
        "has_verify_word": has_verify_word,
        "has_account_word": has_account_word,
        "has_security_word": has_security_word,
        "has_payment_word": has_payment_word,
        "has_bank_word": has_bank_word,

        "url_entropy": url_entropy,
        "domain_entropy": domain_entropy,
        "url_digit_ratio": url_digit_ratio,
        "letter_ratio": letter_ratio_value,
        "special_character_ratio": special_ratio,

        "repeated_characters": repeated_characters,

        "suspicious_word_in_domain": suspicious_word_in_domain,
        "suspicious_word_in_path": suspicious_word_in_path,
        "trusted_tld_with_path_keywords": trusted_tld_with_path_keywords,
        "domain_has_brand_name": domain_has_brand_name,
        "path_to_domain_length_ratio": path_to_domain_length_ratio,
    }

    return features



def main():

    print("=" * 70)
    print("PHISHING URL FEATURE EXTRACTION")
    print("=" * 70)

    print("\nLoading dataset:")

    df = pd.read_csv(INPUT_PATH)

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        df.columns.tolist()
    )


    if "URL" not in df.columns:

        raise ValueError(
            "Dataset must contain a column named 'URL'."
        )

    if "label" not in df.columns:

        raise ValueError(
            "Dataset must contain a column named 'label'."
        )


    print("\nExtracting features...")

    feature_rows = []
    urls = []
    domains = []

    total = len(df)

    for index, url in enumerate(
        df["URL"],
        start=1
    ):

        feature_rows.append(
            extract_features(url)
        )
        
        cleaned_url = clean_input_url(url)
        parsed, hostname = parse_url(cleaned_url)
        feature_hostname = hostname
        if feature_hostname.startswith("www."):
            feature_hostname = feature_hostname[4:]
            
        urls.append(url)
        domains.append(feature_hostname)

        if index % 10000 == 0:

            print(
                f"Processed {index:,}/{total:,}"
            )


    features_df = pd.DataFrame(
        feature_rows
    )
    
    features_df.insert(0, "url", urls)
    features_df.insert(1, "domain", domains)

    features_df["label"] = (
        df["label"]
        .astype(int)
        .values
    )


    
    numeric_cols = features_df.select_dtypes(include=['number']).columns

    features_df[numeric_cols] = features_df[numeric_cols].replace(
        [float("inf"), float("-inf")],
        0
    )

    features_df[numeric_cols] = features_df[numeric_cols].fillna(0)


    features_df.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COMPLETED")
    print("=" * 70)

    print(
        "\nFeature rows:",
        len(features_df)
    )

    print(
        "Feature count:",
        len(features_df.columns) - 3
    )

    print(
        "\nFeatures:"
    )

    for column in features_df.columns:

        if column not in ("label", "url", "domain"):

            print(
                " -",
                column
            )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()