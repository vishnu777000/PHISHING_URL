# Phishing URL Detector

## Problem Statement

This project classifies a URL as benign or potentially phishing using machine-learning models and URL text/structure alone. It does not visit or inspect the website.

## Objectives

- Clean and reproducibly split a labeled URL dataset without URL or registered-domain leakage.
- Extract a stable feature vector that separates hostname, registered domain, subdomain, path, query, and fragment structure.
- Compare Logistic Regression, Random Forest, and linear SVM using held-out evaluation.
- Provide a small command-line prediction interface.

## Dataset

The original CSV is `dataset/phising_urls.csv` and is treated as raw input. Cleaning and generated files are written to `dataset/processed/`. Labels are encoded as `0 = benign` and `1 = phishing`. A deterministic set of 249 benign URL variants is added across multiple university domains and common legitimate URL forms; it is not used to duplicate the whole dataset or add phishing test fixtures.

Current processed data contains 235,145 records: 135,098 benign and 100,047 phishing. The reproducible grouped split contains 141,087 training, 47,028 validation, and 47,030 test records. Registered domains and exact URLs do not overlap between splits.

## Preprocessing

The cleaner canonicalizes HTTP(S) URLs with `urllib.parse`, normalizes supported label names, removes missing or invalid records, removes exact duplicates, and excludes every URL group with conflicting labels. It never overwrites the raw CSV. `tldextract` identifies registered domains locally without network lookups.

## Feature Extraction

The extractor uses URL lengths, hostname/domain/subdomain/path/query structure, character counts by URL component, HTTPS and IP presence, userinfo (`@`), port, punycode, query parameters, entropy, and registered-domain/TLD characteristics. `www` is omitted from structural length and character measurements. Individual path words such as `login`, `verify`, `account`, and `admissions` are not classifier features.

## Algorithms and Training

The training script fits Logistic Regression, Random Forest, and linear SVM. Scaling is included in the Logistic Regression and SVM pipelines. The validation set selects the model by F1, breaking ties by fewer false positives and then fewer false negatives. The test set is used only for final reporting. Default decision cutoffs are retained; no test URL is used as a training target.

## Evaluation

For the current held-out test split:

| Model | Accuracy | Precision | Recall | F1 | False positives | False negatives |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9848 | 0.9882 | 0.9759 | 0.9820 | 233 | 482 |
| Random Forest | 0.9846 | 0.9862 | 0.9774 | 0.9818 | 273 | 452 |
| Linear SVM | 0.9838 | 0.9876 | 0.9742 | 0.9808 | 245 | 517 |

The current validation-selected model is Logistic Regression. Its test confusion matrix, with rows/columns ordered benign then phishing, is `[[26787, 233], [482, 19528]]`. Full classification reports and confusion matrices are generated in `analysis/evaluation_report.txt`; the comparison table and feature-importance/coefficient table are under `analysis/`.

Metrics describe this dataset and split, not guaranteed performance on future URLs. Feature importance for Logistic Regression is based on absolute standardized coefficients and is not causal evidence.

## Installation

Requires Python 3.10 or later. From PowerShell in the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Rebuild the Dataset and Model

Run from the project root in this order:

```powershell
python training/clean_dataset.py
python training/augment_benign.py
python training/feature_extraction.py
python training/split_dataset.py
python training/train_final_model.py
```

Generated splits are stored in `dataset/processed/`; the selected estimator and exact feature configuration are stored under `model/`; evaluation outputs are stored under `analysis/`.

## Command-Line Usage

Interactive mode:

```powershell
python main.py
```

Single URL:

```powershell
python main.py --url "https://www.google.com/search?q=machine+learning"
```

CSV batch input requires a `url` or `URL` column:

```powershell
python main.py --file dataset/processed/test.csv --output predictions.csv
```

Type `exit` or `quit` in interactive mode to stop. The CLI reports probabilities when the selected estimator supports them.
Each prediction is displayed in a bordered summary with the URL, prediction, available probability or margin, and decision threshold.

## Sample Predictions

Representative URLs should be checked after retraining with the commands above. Sanity examples are never appended to the training data solely to force expected results; model predictions may differ as the source dataset or environment changes.

## Limitations

This is a lexical URL classifier, not a real-time security service. It does not use DNS/WHOIS, TLS certificates, website content, crawling, browser integration, or threat-intelligence APIs. A benign-looking prediction is not proof that a site is safe, and suspicious structural patterns can appear on legitimate sites.

## Future Scope

Future work could evaluate on a separately collected, time-separated dataset, examine false positives by URL category, and compare model stability as the dataset changes. Any added analysis should remain URL-based unless the project scope is deliberately changed.