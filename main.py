import sys
import argparse
import json
import pickle
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / 'training'))

try:
    from feature_extraction import extract_features
except ImportError:
    print("Error: Could not import 'extract_features' from 'training.feature_extraction'.")
    sys.exit(1)

def load_model_data():
    models_dir = BASE_DIR / "models"
    model_dir = BASE_DIR / "model"
    
    if (models_dir / "phishing_model_final.pkl").exists():
        model_path = models_dir / "phishing_model_final.pkl"
    elif (model_dir / "phishing_model.pkl").exists():
        model_path = model_dir / "phishing_model.pkl"
    else:
        print("Error: Model file not found.")
        sys.exit(1)
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    if (models_dir / "feature_names_final.pkl").exists():
        fn_path = models_dir / "feature_names_final.pkl"
    elif (model_dir / "feature_names.pkl").exists():
        fn_path = model_dir / "feature_names.pkl"
    else:
        print("Error: Feature names file not found.")
        sys.exit(1)
        
    with open(fn_path, 'rb') as f:
        feature_names = pickle.load(f)

    thresholds = {
        'threshold': 0.50,
        'high_risk_threshold': 0.80,
        'suspicious_threshold': 0.30
    }
    
    threshold_path = models_dir / "threshold.json"
    if threshold_path.exists():
        with open(threshold_path, 'r') as f:
            data = json.load(f)
            thresholds.update(data)
            
    return model, feature_names, thresholds

def get_feature_importances(model, feature_names):
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = abs(model.coef_[0])
    
    if importances is not None:
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        if importance_df['importance'].sum() > 0:
            importance_df['importance'] = (importance_df['importance'] / importance_df['importance'].sum()) * 100
        return importance_df
    return None

def predict_url(url, model, feature_names, thresholds, importances_df=None):
    try:
        features = extract_features(url)
    except Exception as e:
        return {"error": f"Error extracting features: {e}"}
        
    df = pd.DataFrame([features])
    df = df.reindex(columns=feature_names, fill_value=0)
    
    try:
        proba = model.predict_proba(df)[0]
    except Exception as e:
        return {"error": f"Error predicting: {e}"}
        
    benign_prob = proba[0]
    phishing_prob = proba[1] if len(proba) > 1 else 0.0
    
    t = thresholds.get('threshold', 0.50)
    ht = thresholds.get('high_risk_threshold', 0.80)
    st = thresholds.get('suspicious_threshold', 0.30)
    
    if phishing_prob >= t and phishing_prob >= ht:
        pred = "PHISHING"
        risk = "HIGH"
        confidence = phishing_prob
    elif phishing_prob >= t and phishing_prob < ht:
        pred = "PHISHING"
        risk = "MEDIUM"
        confidence = phishing_prob
    elif phishing_prob < t and phishing_prob >= st:
        pred = "SUSPICIOUS"
        risk = "MEDIUM"
        confidence = 1.0 - phishing_prob
    else:
        pred = "BENIGN"
        risk = "LOW"
        confidence = benign_prob
        
    top_features = []
    if importances_df is not None:
        for _, row in importances_df.head(5).iterrows():
            feat = row['feature']
            imp = row['importance']
            val = features.get(feat, 0)
            top_features.append((feat, val, imp))
            
    return {
        "url": url,
        "prediction": pred,
        "risk": risk,
        "confidence": confidence,
        "benign_prob": benign_prob,
        "phishing_prob": phishing_prob,
        "top_features": top_features
    }

def print_single_result(res):
    if "error" in res:
        print(f"Error processing URL: {res['error']}")
        return
        
    print(f"\n{'═'*62}")
    print(f"  PHISHING URL DETECTOR")
    print(f"{'═'*62}\n")
    print(f"  URL:                 {res['url']}")
    print(f"  Prediction:          {res['prediction']}")
    print(f"  Confidence:          {res['confidence']*100:.1f}%")
    print(f"  Risk:                {res['risk']}\n")
    print(f"  Benign Probability:  {res['benign_prob']*100:.1f}%")
    print(f"  Phishing Probability: {res['phishing_prob']*100:.1f}%\n")
    
    if res['top_features']:
        print("  Top Risk Indicators:")
        for i, (feat, val, imp) in enumerate(res['top_features'], 1):
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            print(f"    {i}. {feat} = {val_str:<10} (importance: {imp:.1f}%)")
            
    print(f"\n{'═'*62}\n")

def process_batch(file_path, output_path, model, feature_names, thresholds, importances_df):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
        
    url_col = None
    if 'url' in df.columns:
        url_col = 'url'
    elif 'URL' in df.columns:
        url_col = 'URL'
    else:
        print("Error: CSV must contain a 'url' or 'URL' column.")
        return
        
    urls = df[url_col].dropna().tolist()
    if not urls:
        print("No URLs found to process.")
        return
        
    print(f"Processing {len(urls)} URLs...\n")
    
    results = []
    counts = {"BENIGN": 0, "PHISHING": 0, "SUSPICIOUS": 0, "ERROR": 0}
    
    for url in urls:
        if not str(url).strip():
            continue
        res = predict_url(url, model, feature_names, thresholds, importances_df)
        if "error" in res:
            print(f"[ERROR] {url} - {res['error']}")
            counts["ERROR"] += 1
            results.append({
                "url": url,
                "prediction": "ERROR",
                "confidence": 0,
                "risk": "UNKNOWN",
                "benign_prob": 0,
                "phishing_prob": 0
            })
            continue
            
        print(f"[{res['prediction']:^10}] ({res['confidence']*100:5.1f}%) {url}")
        counts[res['prediction']] += 1
        
        results.append({
            "url": url,
            "prediction": res['prediction'],
            "confidence": res['confidence'],
            "risk": res['risk'],
            "benign_prob": res['benign_prob'],
            "phishing_prob": res['phishing_prob']
        })
        
    print(f"\nSummary:")
    print(f"Total:      {len(urls)}")
    print(f"Benign:     {counts['BENIGN']}")
    print(f"Phishing:   {counts['PHISHING']}")
    print(f"Suspicious: {counts['SUSPICIOUS']}")
    if counts["ERROR"] > 0:
        print(f"Errors:     {counts['ERROR']}")
        
    if output_path:
        out_df = pd.DataFrame(results)
        try:
            out_df.to_csv(output_path, index=False)
            print(f"\nResults saved to {output_path}")
        except Exception as e:
            print(f"Error saving results: {e}")

def interactive_mode(model, feature_names, thresholds, importances_df):
    print(f"\n{'═'*62}")
    print(f"  PHISHING URL DETECTOR - INTERACTIVE MODE")
    print(f"  Type 'exit' or 'quit' to stop.")
    print(f"{'═'*62}\n")
    
    while True:
        try:
            url = input("Enter URL to scan: ").strip()
            if not url:
                continue
            if url.lower() in ['exit', 'quit']:
                break
            
            res = predict_url(url, model, feature_names, thresholds, importances_df)
            print_single_result(res)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Phishing URL Detector Predictor")
    parser.add_argument("--url", type=str, help="Scan a single URL")
    parser.add_argument("--file", type=str, help="Scan all URLs from a CSV file")
    parser.add_argument("--output", type=str, help="Save batch results to CSV")
    
    args = parser.parse_args()
    
    model, feature_names, thresholds = load_model_data()
    importances_df = get_feature_importances(model, feature_names)
    
    if args.url:
        res = predict_url(args.url, model, feature_names, thresholds, importances_df)
        print_single_result(res)
    elif args.file:
        process_batch(args.file, args.output, model, feature_names, thresholds, importances_df)
    else:
        interactive_mode(model, feature_names, thresholds, importances_df)

if __name__ == '__main__':
    main()
