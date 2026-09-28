# Phishing URL Detector

A machine-learning project that extracts URL characteristics and classifies URLs as benign or potentially phishing. It includes a Flask web interface, a command-line scanner, dataset preparation scripts, and model training/evaluation utilities.

## Requirements

- Python 3.10 or later
- pip

Install the dependencies from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation for this session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Run the Web App

From the project root with the virtual environment activated:

```powershell
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser. Enter a URL and select **Scan URL**. Stop the local server with `Ctrl+C`.

The Flask app loads the trained model and feature-name list from `model/` at startup. Keep both model files available when running the app.

## Run the Command-Line Scanner

Scan one URL:

```powershell
python main.py --url "https://example.com"
```

Scan URLs from a CSV and save the results:

```powershell
python main.py --file dataset/test.csv --output results.csv
```

Run without arguments to use interactive mode:

```powershell
python main.py
```

The batch CSV needs a `url` or `URL` column.

## Project Layout

- `app.py`: Flask web application and prediction endpoint
- `main.py`: command-line and batch prediction tool
- `training/feature_extraction.py`: URL parsing and feature extraction
- `training/`: data preparation, model training, evaluation, and analysis scripts
- `dataset/`: URL datasets and extracted feature/split files
- `model/`: trained model and feature-name artifact used by the app
- `analysis/`: model-analysis outputs
- `requirements.txt`: Python dependencies

## Training and Evaluation

The scripts in `training/` cover dataset cleaning, augmentation, feature extraction, dataset splitting, model comparison, training, and evaluation. Review each script's inputs and outputs before running it, since some scripts regenerate files under `dataset/`, `analysis/`, or the project root.

## Important Limitations

This is a machine-learning aid, not a guarantee that a URL is safe or malicious. Verify suspicious links independently and do not enter credentials based only on this tool's result. The current saved model has known false-positive behavior for some legitimate university URLs, including the Maya portal's `/sign-in` path. Model predictions can also depend on the exact URL form and trained feature set.
