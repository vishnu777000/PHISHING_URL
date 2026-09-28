from pathlib import Path
import re

import joblib
import pandas as pd

from feature_extraction import extract_features



BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "phishing_model.pkl"
)

FEATURE_PATH = (
    BASE_DIR
    / "model"
    / "feature_names.pkl"
)



def clean_input_url(url):

    url = str(url).strip()


    match = re.match(
        r"^\[.*?\]\((https?://.*?)\)$",
        url
    )

    if match:
        url = match.group(1)

    url = url.strip("\"'")

    return url.strip()



print("=" * 60)
print("LOADING PHISHING URL MODEL")
print("=" * 60)

model = joblib.load(
    MODEL_PATH
)

feature_names = joblib.load(
    FEATURE_PATH
)

print("Model loaded successfully.")



def predict_url(url):

    url = clean_input_url(url)

    features = extract_features(url)

    row = pd.DataFrame(
        [features]
    )

    row = row.reindex(
        columns=feature_names
    )

    prediction = int(
        model.predict(row)[0]
    )

    probabilities = (
        model.predict_proba(row)[0]
    )

    return (
        url,
        prediction,
        probabilities
    )



while True:

    url = input(
        "\nEnter URL to test "
        "(or type exit): "
    ).strip()

    if url.lower() == "exit":

        print(
            "\nExiting..."
        )

        break

    if not url:

        print(
            "Please enter a URL."
        )

        continue

    try:

        clean_url, prediction, probabilities = (
            predict_url(url)
        )

        benign_probability = (
            probabilities[0] * 100
        )

        phishing_probability = (
            probabilities[1] * 100
        )


        print(
            "\n"
            + "=" * 60
        )

        print(
            "URL:",
            clean_url
        )


        if prediction == 1:

            print(
                "Prediction: PHISHING"
            )

        else:

            print(
                "Prediction: BENIGN"
            )


        print(
            f"Benign probability  : "
            f"{benign_probability:.2f}%"
        )

        print(
            f"Phishing probability: "
            f"{phishing_probability:.2f}%"
        )


        print(
            "=" * 60
        )


    except Exception as e:

        print(
            "\nERROR:"
        )

        print(e)