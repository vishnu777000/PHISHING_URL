from pathlib import Path
import json
import re

import joblib
import pandas as pd

from feature_extraction import extract_features


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_CANDIDATES = [
    BASE_DIR / "model" / "phishing_model.pkl",
    BASE_DIR / "model" / "phishing_model_final.joblib",
    BASE_DIR / "models" / "phishing_model_final.pkl",
    BASE_DIR / "models" / "phishing_model.pkl",
]

FEATURE_CANDIDATES = [
    BASE_DIR / "model" / "feature_names.pkl",
    BASE_DIR / "models" / "feature_names_final.pkl",
    BASE_DIR / "models" / "feature_names.pkl",
    BASE_DIR / "model" / "model_metadata.json",
]


def resolve_first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def clean_input_url(url):
    url = str(url).strip()

    match = re.match(r"^\[.*?\]\((https?://.*?)\)$", url)
    if match:
        url = match.group(1)

    url = url.strip("\"'")
    return url.strip()


def load_metadata_feature_names():
    metadata_path = BASE_DIR / "model" / "model_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        feature_names = metadata.get("feature_names")
        if feature_names:
            return feature_names

    feature_path = resolve_first_existing(FEATURE_CANDIDATES)
    if feature_path.suffix == ".json":
        return json.loads(feature_path.read_text(encoding="utf-8")).get("feature_names")
    return joblib.load(feature_path)


def load_model_and_features():
    model_path = resolve_first_existing(MODEL_CANDIDATES)
    feature_names = load_metadata_feature_names()

    if model_path.exists():
        model = joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model file not found. Checked: {MODEL_CANDIDATES}")

    if feature_names is None:
        raise FileNotFoundError(f"Feature names not found. Checked: {FEATURE_CANDIDATES}")

    return model, feature_names


def predict_url(url, model, feature_names):
    url = clean_input_url(url)
    features = extract_features(url)
    row = pd.DataFrame([features]).reindex(columns=feature_names)
    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    return url, prediction, probabilities


def main():
    print("=" * 60)
    print("LOADING PHISHING URL MODEL")
    print("=" * 60)

    try:
        model, feature_names = load_model_and_features()
    except FileNotFoundError as exc:
        print(f"Model load failed: {exc}")
        raise SystemExit(1)

    print("Model loaded successfully.")

    while True:
        url = input("\nEnter URL to test (or type exit): ").strip()

        if url.lower() == "exit":
            print("\nExiting...")
            break

        if not url:
            print("Please enter a URL.")
            continue

        try:
            clean_url, prediction, probabilities = predict_url(url, model, feature_names)
            benign_probability = probabilities[0] * 100
            phishing_probability = probabilities[1] * 100

            print("\n" + "=" * 60)
            print("URL:", clean_url)
            print("Prediction: PHISHING" if prediction == 1 else "Prediction: BENIGN")
            print(f"Benign probability  : {benign_probability:.2f}%")
            print(f"Phishing probability: {phishing_probability:.2f}%")
            print("=" * 60)
        except Exception as exc:
            print("\nERROR:")
            print(exc)


def test_model_can_load_and_predict():
    model, feature_names = load_model_and_features()
    assert model is not None
    assert len(feature_names) > 0

    url, prediction, probabilities = predict_url("https://example.com", model, feature_names)
    assert url == "https://example.com"
    assert prediction in {0, 1}
    assert len(probabilities) == 2


if __name__ == "__main__":
    main()