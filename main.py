import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "phishing_model_final.joblib"
METADATA_PATH = BASE_DIR / "model" / "model_metadata.json"
OUTPUT_BORDER = "=" * 60
sys.path.insert(0, str(BASE_DIR / "training"))
from feature_extraction import extract_features


def load_model():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Trained model artifacts are missing. Run: python training/train_final_model.py"
        )

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    feature_names = metadata.get("feature_names", [])
    current_features = list(extract_features("example.com"))
    if feature_names != current_features:
        raise RuntimeError("Saved model feature order does not match the current URL extractor.")
    return model, metadata


def predict_url(url, model, metadata):
    features = extract_features(url)
    feature_names = metadata["feature_names"]
    if list(features) != feature_names:
        raise RuntimeError("Prediction feature names or order do not match the saved model.")

    feature_row = pd.DataFrame([[features[name] for name in feature_names],], columns=feature_names)
    cutoff = float(metadata["decision_cutoff"])
    has_probabilities = hasattr(model, "predict_proba")
    if has_probabilities:
        class_index = list(model.classes_).index(1)
        phishing_score = float(model.predict_proba(feature_row)[0][class_index])
    else:
        positive_index = list(model.classes_).index(1)
        margin = float(model.decision_function(feature_row)[0])
        phishing_score = margin if positive_index == 1 else -margin

    prediction = "PHISHING" if phishing_score >= cutoff else "BENIGN"
    return {
        "url": url,
        "prediction": prediction,
        "score": phishing_score,
        "score_name": "phishing_probability" if has_probabilities else "phishing_margin",
        "has_probabilities": has_probabilities,
        "cutoff": cutoff,
    }


def print_result(result):
    print(OUTPUT_BORDER)
    print("PHISHING URL DETECTOR")
    print(OUTPUT_BORDER)
    print()
    print("URL:")
    print(result["url"])
    print()
    print(f"Prediction: {result['prediction']}")
    if result["has_probabilities"]:
        print(f"Benign probability: {(1 - result['score']) * 100:.2f}%")
        print(f"Phishing probability: {result['score'] * 100:.2f}%")
    else:
        print(f"Phishing decision margin: {result['score']:.4f}")
    print(f"Decision threshold: {result['cutoff']:.2f}")
    print(OUTPUT_BORDER)


def scan_file(input_file, output_file, model, metadata):
    frame = pd.read_csv(input_file)
    url_column = next((column for column in ("URL", "url") if column in frame), None)
    if url_column is None:
        raise ValueError("CSV input must contain a URL or url column.")

    results = []
    for url in frame[url_column]:
        try:
            results.append(predict_url(url, model, metadata))
        except (TypeError, ValueError, RuntimeError) as error:
            results.append({"url": url, "prediction": "ERROR", "error": str(error)})

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")


def interactive_mode(model, metadata):
    print("Type 'exit' to quit.")
    while True:
        try:
            url = input("\nEnter URL: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if url.lower() in {"exit", "quit"}:
            return
        if not url:
            print("Enter a URL or type 'exit' to quit.")
            continue
        try:
            print_result(predict_url(url, model, metadata))
        except (TypeError, ValueError, RuntimeError) as error:
            print(f"Invalid URL: {error}")


def main():
    parser = argparse.ArgumentParser(description="URL-based phishing classifier")
    parser.add_argument("--url", help="Classify one URL")
    parser.add_argument("--file", help="Classify URLs from a CSV file")
    parser.add_argument("--output", default="prediction_results.csv", help="Batch results CSV path")
    args = parser.parse_args()

    try:
        model, metadata = load_model()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"Startup error: {error}", file=sys.stderr)
        raise SystemExit(1)

    if args.url:
        try:
            print_result(predict_url(args.url, model, metadata))
        except (TypeError, ValueError, RuntimeError) as error:
            print(f"Invalid URL: {error}", file=sys.stderr)
            raise SystemExit(2)
    elif args.file:
        try:
            scan_file(args.file, args.output, model, metadata)
        except (OSError, ValueError) as error:
            print(f"Input error: {error}", file=sys.stderr)
            raise SystemExit(2)
    else:
        interactive_mode(model, metadata)


if __name__ == "__main__":
    main()