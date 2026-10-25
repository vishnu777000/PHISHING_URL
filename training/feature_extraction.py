from collections import Counter
import ipaddress
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

import pandas as pd
import tldextract


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "dataset" / "processed" / "augmented_urls.csv"
OUTPUT_PATH = BASE_DIR / "dataset" / "processed" / "features.csv"
DOMAIN_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())


# ============================================================
# CONSTANTS
# ============================================================

SUSPICIOUS_WORDS = {
    "login", "signin", "sign-in", "log-in",
    "verify", "verification", "validate",
    "account", "update", "secure", "security",
    "confirm", "confirmation",
    "password", "credential", "credentials",
    "bank", "banking",
    "payment", "payments", "wallet",
    "authorize", "authorization",
    "authentication",
    "recover", "recovery",
    "unlock", "suspend", "suspended",
    "billing", "invoice", "transaction",
    "support", "webscr"
}

LOGIN_WORDS = {
    "login", "signin", "sign-in", "log-in"
}

VERIFY_WORDS = {
    "verify", "verification", "confirm",
    "confirmation", "validate"
}

ACCOUNT_WORDS = {
    "account", "credential", "credentials",
    "password"
}

SECURITY_WORDS = {
    "secure", "security", "authentication",
    "authorize", "authorization"
}

PAYMENT_WORDS = {
    "payment", "payments", "billing",
    "invoice", "transaction", "wallet"
}

BANK_WORDS = {
    "bank", "banking"
}


SHORTENING_SERVICES = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl",
    "is.gd", "buff.ly", "ow.ly", "cutt.ly",
    "shorturl.at", "rebrand.ly", "tiny.cc",
    "rb.gy", "lnkd.in"
}


TRUSTED_TLDS = {
    ".edu", ".ac.in", ".edu.in", ".ac.uk",
    ".edu.au", ".ac.jp", ".edu.cn", ".edu.br",
    ".edu.mx", ".ac.nz", ".edu.sg", ".ac.kr",
    ".edu.pk", ".edu.ng", ".edu.za", ".ac.id",
    ".edu.my", ".edu.ph", ".edu.tw", ".edu.hk",
    ".edu.eg", ".edu.tr", ".edu.ar", ".edu.co",
    ".edu.pe", ".ac.th",
    ".gov", ".gov.in", ".gov.uk", ".gov.au",
    ".gov.br", ".go.jp", ".gob.mx", ".gc.ca",
    ".mil"
}


SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq",
    ".xyz", ".top", ".buzz", ".club",
    ".work", ".click", ".link", ".info",
    ".icu", ".rest", ".cam"
}


# Used only as a feature.
# It is NOT used as a whitelist.
BRAND_DOMAINS = {
    "paypal": "paypal.com",
    "apple": "apple.com",
    "microsoft": "microsoft.com",
    "google": "google.com",
    "facebook": "facebook.com",
    "amazon": "amazon.com",
    "netflix": "netflix.com",
    "instagram": "instagram.com",
    "whatsapp": "whatsapp.com",
    "telegram": "telegram.com",
    "linkedin": "linkedin.com",
    "twitter": "twitter.com",
    "chase": "chase.com",
    "wellsfargo": "wellsfargo.com",
    "bankofamerica": "bankofamerica.com",
    "citibank": "citibank.com",
    "americanexpress": "americanexpress.com"
}


# ============================================================
# URL HELPERS
# ============================================================

def clean_input_url(url):
    if url is None or pd.isna(url):
        raise ValueError("URL is missing.")

    url = str(url).strip()

    # Markdown URL
    match = re.match(
        r"^\[.*?\]\((https?://.*?)\)$",
        url
    )

    if match:
        url = match.group(1)

    url = url.strip("\"' ")

    if not url:
        raise ValueError("URL is empty.")

    url = url.removeprefix("//")

    if "://" not in url:
        url = "https://" + url

    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"Invalid URL: {error}") from error

    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        raise ValueError("Enter a valid HTTP or HTTPS URL with a hostname.")

    try:
        hostname = hostname.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError as error:
        raise ValueError("URL hostname is invalid.") from error

    userinfo = ""
    if "@" in parsed.netloc:
        userinfo = parsed.netloc.rsplit("@", 1)[0] + "@"

    netloc = f"{userinfo}{hostname}"
    if port is not None:
        netloc += f":{port}"

    return parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=netloc,
        path=parsed.path or "/",
    ).geturl()


def parse_url(url):
    normalized_url = clean_input_url(url)
    parsed = urlsplit(normalized_url)
    return parsed, (parsed.hostname or "").lower().rstrip(".")


def remove_www(hostname):
    if hostname.startswith("www."):
        return hostname[4:]
    return hostname


# ============================================================
# STATISTICS
# ============================================================

def calculate_entropy(text):
    if not text:
        return 0.0

    counter = Counter(text)
    length = len(text)

    entropy = 0.0

    for count in counter.values():
        p = count / length
        entropy -= p * math.log2(p)

    return entropy


def digit_ratio(text):
    if not text:
        return 0.0

    return sum(c.isdigit() for c in text) / len(text)


def letter_ratio(text):
    if not text:
        return 0.0

    return sum(c.isalpha() for c in text) / len(text)


def special_character_ratio(text):
    if not text:
        return 0.0

    return sum(not c.isalnum() for c in text) / len(text)


def has_repeated_characters(text):
    if not text:
        return 0

    return int(bool(re.search(r"(.)\1\1", text)))


# ============================================================
# DOMAIN FEATURES
# ============================================================

def is_ip_address(hostname):
    if not hostname:
        return 0

    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def has_trusted_tld(hostname):
    hostname = hostname.lower()

    return int(
        any(hostname.endswith(tld) for tld in TRUSTED_TLDS)
    )


def has_suspicious_tld(hostname):
    hostname = hostname.lower()

    return int(
        any(hostname.endswith(tld) for tld in SUSPICIOUS_TLDS)
    )


def count_subdomains(hostname):
    if not hostname:
        return 0

    if is_ip_address(hostname):
        return 0

    parts = [p for p in hostname.split(".") if p]

    # Approximation of subdomain count.
    # Example:
    # google.com -> 0
    # mail.google.com -> 1
    # a.b.google.com -> 2
    return max(0, len(parts) - 2)


def domain_has_brand(hostname):
    """
    Detects suspicious brand names in a domain.

    Example:
        paypal.com -> 0
        paypal-login-security.com -> 1
        secure-paypal.example -> 1

    This is deliberately NOT a trusted-domain whitelist.
    """

    hostname = remove_www(hostname.lower())

    for brand, official_domain in BRAND_DOMAINS.items():

        if brand not in hostname:
            continue

        if hostname == official_domain:
            continue

        if hostname.endswith("." + official_domain):
            continue

        return 1

    return 0


# ============================================================
# WORD FEATURES
# ============================================================

def contains_any_word(text, words):
    text = str(text).lower()

    for word in words:

        pattern = (
            rf"(?<![a-z])"
            rf"{re.escape(word)}"
            rf"(?![a-z])"
        )

        if re.search(pattern, text):
            return 1

    return 0


def suspicious_word_count(text):
    text = str(text).lower()

    count = 0

    for word in SUSPICIOUS_WORDS:

        pattern = (
            rf"(?<![a-z])"
            rf"{re.escape(word)}"
            rf"(?![a-z])"
        )

        if re.search(pattern, text):
            count += 1

    return count


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(url):

    cleaned_url = clean_input_url(url)

    parsed, hostname = parse_url(cleaned_url)

    hostname_no_www = remove_www(hostname)
    extraction = DOMAIN_EXTRACTOR(hostname_no_www)
    registered_domain = extraction.top_domain_under_public_suffix or hostname_no_www
    subdomain = extraction.subdomain

    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""

    # Do not treat "/" alone as meaningful path complexity.
    if path == "/":
        path = ""

    # Ignore scheme and www in lexical measurements so equivalent host forms
    # have the same structural features.
    structural_url = f"{hostname_no_www}{path}"
    if query:
        structural_url += f"?{query}"
    if fragment:
        structural_url += f"#{fragment}"

    url_length = len(structural_url)

    hostname_length = len(hostname_no_www)

    domain_length = len(registered_domain)

    subdomain_length = len(subdomain)

    path_length = len(path)

    query_length = len(query)

    fragment_length = len(fragment)

    # --------------------------------------------------------
    # Character counts
    # --------------------------------------------------------

    num_dots = structural_url.count(".")
    num_hyphens = structural_url.count("-")
    num_underscores = structural_url.count("_")
    num_slashes = path.count("/")
    num_backslashes = structural_url.count("\\")
    num_digits = sum(c.isdigit() for c in structural_url)
    num_letters = sum(c.isalpha() for c in structural_url)

    num_special_chars = sum(
        not c.isalnum()
        for c in structural_url
    )

    num_question_marks = structural_url.count("?")
    num_equals = structural_url.count("=")
    num_ampersands = structural_url.count("&")
    num_percent = structural_url.count("%")
    num_colons = structural_url.count(":")
    num_semicolons = structural_url.count(";")
    num_at = parsed.netloc.count("@")
    hostname_dots = hostname_no_www.count(".")
    hostname_hyphens = hostname_no_www.count("-")
    hostname_digits = sum(character.isdigit() for character in hostname_no_www)
    registered_domain_hyphens = registered_domain.count("-")
    path_hyphens = path.count("-")
    path_digits = sum(character.isdigit() for character in path)
    path_special_characters = sum(not character.isalnum() for character in path)

    # --------------------------------------------------------
    # Domain
    # --------------------------------------------------------

    has_ip = is_ip_address(hostname)

    num_subdomains = 0 if is_ip_address(hostname_no_www) else len(
        [part for part in subdomain.split(".") if part]
    )

    domain_digits = sum(
        c.isdigit()
        for c in registered_domain
    )

    domain_digit_ratio = digit_ratio(
        registered_domain
    )
    subdomain_digits = sum(c.isdigit() for c in subdomain)
    subdomain_hyphens = subdomain.count("-")

    # --------------------------------------------------------
    # Path
    # --------------------------------------------------------

    path_segments = [
        segment
        for segment in path.split("/")
        if segment
    ]

    path_depth = len(path_segments)

    # --------------------------------------------------------
    # Protocol / structural
    # --------------------------------------------------------

    has_https = int(
        parsed.scheme.lower() == "https"
    )

    try:
        has_port = int(parsed.port is not None)
    except ValueError:
        has_port = 1

    has_punycode = int(
        "xn--" in hostname_no_www
    )

    is_shortened = int(
        hostname_no_www in SHORTENING_SERVICES
    )

    is_trusted_tld = has_trusted_tld(
        hostname_no_www
    )

    is_suspicious_tld = has_suspicious_tld(
        hostname_no_www
    )

    has_at = int("@" in parsed.netloc)

    has_question_mark = int(bool(query))

    has_fragment = int(bool(fragment))

    has_double_slash = int("//" in path)
    has_userinfo = int("@" in parsed.netloc)
    query_param_count = len([part for part in query.split("&") if part])

    # --------------------------------------------------------
    # Ratios
    # --------------------------------------------------------

    url_entropy = calculate_entropy(structural_url)

    domain_entropy = calculate_entropy(
        registered_domain
    )
    subdomain_entropy = calculate_entropy(subdomain)
    path_entropy = calculate_entropy(path)
    query_entropy = calculate_entropy(query)

    url_digit_ratio = digit_ratio(structural_url)

    url_letter_ratio = letter_ratio(structural_url)

    special_ratio = special_character_ratio(structural_url)

    repeated_characters = has_repeated_characters(structural_url)

    path_to_domain_length_ratio = (
        path_length / max(domain_length, 1)
    )

    domain_has_brand_name = domain_has_brand(
        registered_domain
    )

    # ========================================================
    # EXACT FEATURE ORDER
    # ========================================================

    features = {

        "url_length": url_length,
        "hostname_length": hostname_length,
        "domain_length": domain_length,
        "subdomain_length": subdomain_length,
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
        "hostname_dots": hostname_dots,
        "hostname_hyphens": hostname_hyphens,
        "hostname_digits": hostname_digits,
        "registered_domain_hyphens": registered_domain_hyphens,
        "path_hyphens": path_hyphens,
        "path_digits": path_digits,
        "path_special_characters": path_special_characters,

        "num_subdomains": num_subdomains,
        "domain_digits": domain_digits,
        "domain_digit_ratio": domain_digit_ratio,
        "subdomain_digits": subdomain_digits,
        "subdomain_hyphens": subdomain_hyphens,

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
        "has_userinfo": has_userinfo,
        "query_param_count": query_param_count,

        "url_entropy": url_entropy,
        "domain_entropy": domain_entropy,
        "subdomain_entropy": subdomain_entropy,
        "path_entropy": path_entropy,
        "query_entropy": query_entropy,
        "url_digit_ratio": url_digit_ratio,
        "letter_ratio": url_letter_ratio,
        "special_character_ratio": special_ratio,

        "repeated_characters": repeated_characters,

        "domain_has_brand_name": domain_has_brand_name,
        "path_to_domain_length_ratio": path_to_domain_length_ratio,
    }

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHISHING URL FEATURE EXTRACTION")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    if "URL" not in df.columns:
        raise ValueError(
            "augmented_urls.csv must contain 'URL'"
        )

    if "label" not in df.columns:
        raise ValueError(
            "augmented_urls.csv must contain 'label'"
        )

    feature_rows = []
    urls = []
    domains = []
    valid_labels = []

    total = len(df)

    for i, row in df.iterrows():

        url = row["URL"]

        try:
            features = extract_features(url)
            _, hostname = parse_url(url)

            feature_rows.append(features)
            urls.append(url)
            domains.append(remove_www(hostname))
            valid_labels.append(int(row["label"]))

        except ValueError as e:

            print(f"[WARNING] Skipping invalid URL: {url}")
            print(f"          {e}")

        if (i + 1) % 10000 == 0:
            print(
                f"Processed {i + 1:,}/{total:,}"
            )

    features_df = pd.DataFrame(feature_rows)

    features_df.insert(0, "url", urls)
    features_df.insert(1, "domain", domains)
    features_df["label"] = valid_labels

    numeric_columns = features_df.select_dtypes(
        include=["number"]
    ).columns

    features_df[numeric_columns] = (
        features_df[numeric_columns]
        .replace([float("inf"), float("-inf")], 0)
        .fillna(0)
    )

    feature_columns = [
        c for c in features_df.columns
        if c not in {"url", "domain", "label"}
    ]

    expected_features = list(extract_features("example.com"))
    if feature_columns != expected_features:
        raise RuntimeError("Feature output order differs from the inference schema.")

    features_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(f"Rows: {len(features_df):,}")
    print(f"Features: {len(feature_columns)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()