"""
Phishing URL Detector — Flask Web Application

Loads the trained Random Forest model and provides a modern web UI
for scanning URLs and displaying phishing/safe predictions.
"""

import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = BASE_DIR / "training"

sys.path.insert(0, str(TRAINING_DIR))


import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

from feature_extraction import extract_features


app = Flask(__name__)


MODEL_PATH = BASE_DIR / "model" / "phishing_model.pkl"
FEATURE_PATH = BASE_DIR / "model" / "feature_names.pkl"

print("Loading model …")
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)
print(f"Model loaded — expects {len(feature_names)} features.")



def predict_url(url: str) -> dict:
    """Extract features, run model, return a result dict."""

    PHISHING_THRESHOLD = 0.55

    features = extract_features(url)

    row = pd.DataFrame([features]).reindex(columns=feature_names, fill_value=0)

    probabilities = model.predict_proba(row)[0]
    phishing_prob = probabilities[1]

    prediction = int(phishing_prob >= PHISHING_THRESHOLD)
    confidence = probabilities[prediction] * 100

    if prediction == 1:
        label = "Phishing"
        if confidence >= 90:
            risk = "High Risk"
        elif confidence >= 70:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"
    else:
        label = "Safe"
        risk = "Low Risk"

    importances = model.feature_importances_
    top_indices = importances.argsort()[::-1][:5]
    top_features = [
        {
            "name": feature_names[i],
            "importance": round(float(importances[i]) * 100, 2),
            "value": round(float(row.iloc[0, i]), 4),
        }
        for i in top_indices
    ]

    return {
        "url": url,
        "prediction": label,
        "is_phishing": prediction == 1,
        "confidence": round(confidence, 2),
        "risk": risk,
        "probabilities": {
            "safe": round(float(probabilities[0]) * 100, 2),
            "phishing": round(float(probabilities[1]) * 100, 2),
        },
        "top_features": top_features,
    }



@app.route("/")
def index():
    """Serve the single-page UI."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    """Accept a URL and return the prediction as JSON."""
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()

    if not url:
        return jsonify({"error": "Please provide a URL to scan."}), 400

    try:
        result = predict_url(url)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500



HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Phishing URL Detector</title>
<style>
  
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem 1rem;
  }

  
  .header {
    text-align: center;
    margin-bottom: 2rem;
  }
  .header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d2ff, #3a7bd5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: .4rem;
  }
  .header p {
    color: #9a9abf;
    font-size: .95rem;
  }

  
  .card {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 2rem;
    width: 100%;
    max-width: 580px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, .35);
  }

  
  .input-group {
    display: flex;
    gap: .6rem;
  }
  .input-group input {
    flex: 1;
    padding: .85rem 1rem;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,.12);
    background: rgba(255,255,255,.07);
    color: #fff;
    font-size: 1rem;
    outline: none;
    transition: border .2s;
  }
  .input-group input::placeholder { color: #777; }
  .input-group input:focus { border-color: #3a7bd5; }

  .btn {
    padding: .85rem 1.6rem;
    border: none;
    border-radius: 12px;
    background: linear-gradient(135deg, #3a7bd5, #00d2ff);
    color: #fff;
    font-weight: 600;
    font-size: .95rem;
    cursor: pointer;
    transition: opacity .2s, transform .1s;
    white-space: nowrap;
  }
  .btn:hover { opacity: .9; }
  .btn:active { transform: scale(.97); }
  .btn:disabled {
    opacity: .5;
    cursor: not-allowed;
  }

  
  .spinner {
    display: none;
    text-align: center;
    padding: 2rem 0;
  }
  .spinner.active { display: block; }
  .spinner .dot {
    display: inline-block;
    width: 10px; height: 10px;
    margin: 0 4px;
    border-radius: 50%;
    background: #3a7bd5;
    animation: bounce .6s infinite alternate;
  }
  .spinner .dot:nth-child(2) { animation-delay: .2s; }
  .spinner .dot:nth-child(3) { animation-delay: .4s; }
  @keyframes bounce { to { transform: translateY(-12px); opacity: .4; } }

  
  #result {
    display: none;
    animation: fadeUp .4s ease;
  }
  #result.show { display: block; }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .verdict {
    text-align: center;
    padding: 1.4rem 0;
  }
  .verdict .icon { font-size: 3rem; margin-bottom: .4rem; }
  .verdict .label {
    font-size: 1.6rem;
    font-weight: 700;
  }
  .verdict .confidence {
    font-size: .95rem;
    margin-top: .3rem;
    color: #bbb;
  }
  .verdict.safe .label  { color: #2ecc71; }
  .verdict.phish .label { color: #e74c3c; }

  
  .risk-bar-container {
    margin: 1.2rem 0;
  }
  .risk-bar-container .meta {
    display: flex;
    justify-content: space-between;
    font-size: .82rem;
    color: #999;
    margin-bottom: .35rem;
  }
  .risk-bar {
    height: 8px;
    border-radius: 6px;
    background: rgba(255,255,255,.08);
    overflow: hidden;
  }
  .risk-bar .fill {
    height: 100%;
    border-radius: 6px;
    transition: width .6s ease;
  }

  
  .probs {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.2rem;
  }
  .prob-box {
    flex: 1;
    text-align: center;
    padding: .8rem;
    border-radius: 12px;
    background: rgba(255,255,255,.04);
  }
  .prob-box .value { font-size: 1.3rem; font-weight: 700; }
  .prob-box .tag   { font-size: .75rem; color: #999; margin-top: .2rem; }
  .prob-box.safe  .value { color: #2ecc71; }
  .prob-box.phish .value { color: #e74c3c; }

  
  .features-section h3 {
    font-size: .9rem;
    color: #888;
    margin-bottom: .6rem;
  }
  .feature-table {
    width: 100%;
    border-collapse: collapse;
    font-size: .85rem;
  }
  .feature-table th,
  .feature-table td {
    text-align: left;
    padding: .45rem .6rem;
    border-bottom: 1px solid rgba(255,255,255,.06);
  }
  .feature-table th { color: #777; font-weight: 500; }
  .feature-table td:last-child { text-align: right; color: #aaa; }

  
  .error-msg {
    display: none;
    text-align: center;
    color: #e74c3c;
    padding: 1rem 0;
    font-size: .95rem;
  }
  .error-msg.show { display: block; }

  
  .footer {
    margin-top: auto;
    padding-top: 2rem;
    font-size: .78rem;
    color: #555;
    text-align: center;
  }

  
  @media (max-width: 480px) {
    .input-group { flex-direction: column; }
    .header h1 { font-size: 1.6rem; }
    .probs { flex-direction: column; }
  }
</style>
</head>
<body>

  <div class="header">
    <h1>🛡️ Phishing URL Detector</h1>
    <p>Enter any URL below to check if it's safe or a phishing attempt</p>
  </div>

  
  <div class="card">
    <form id="scanForm" autocomplete="off">
      <div class="input-group">
        <input
          id="urlInput"
          type="text"
          placeholder="https://example.com/login"
          autofocus
          required
        />
        <button class="btn" type="submit" id="scanBtn">Scan URL</button>
      </div>
    </form>
  </div>

  
  <div class="card" id="resultCard" style="display:none">

    
    <div class="spinner" id="spinner">
      <div class="dot"></div><div class="dot"></div><div class="dot"></div>
    </div>

    
    <div class="error-msg" id="errorMsg"></div>

    
    <div id="result">

      <div class="verdict" id="verdict">
        <div class="icon" id="verdictIcon"></div>
        <div class="label" id="verdictLabel"></div>
        <div class="confidence" id="verdictConf"></div>
      </div>

      <div class="risk-bar-container">
        <div class="meta">
          <span>Risk Level</span>
          <span id="riskLabel"></span>
        </div>
        <div class="risk-bar">
          <div class="fill" id="riskFill"></div>
        </div>
      </div>

      <div class="probs">
        <div class="prob-box safe">
          <div class="value" id="safeProb">—</div>
          <div class="tag">Safe</div>
        </div>
        <div class="prob-box phish">
          <div class="value" id="phishProb">—</div>
          <div class="tag">Phishing</div>
        </div>
      </div>

      <div class="features-section">
        <h3>Top Contributing Features</h3>
        <table class="feature-table">
          <thead>
            <tr><th>Feature</th><th>Value</th><th>Importance</th></tr>
          </thead>
          <tbody id="featureBody"></tbody>
        </table>
      </div>

    </div>
  </div>

  <div class="footer">
    Powered by a Random Forest model &middot; ML-based URL analysis
  </div>

<script>
const form      = document.getElementById('scanForm');
const urlInput  = document.getElementById('urlInput');
const scanBtn   = document.getElementById('scanBtn');
const resultCard= document.getElementById('resultCard');
const spinner   = document.getElementById('spinner');
const errorMsg  = document.getElementById('errorMsg');
const result    = document.getElementById('result');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const url = urlInput.value.trim();
  if (!url) return;

  resultCard.style.display = 'block';
  spinner.classList.add('active');
  result.classList.remove('show');
  errorMsg.classList.remove('show');
  scanBtn.disabled = true;
  scanBtn.textContent = 'Scanning…';

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });

    const data = await res.json();

    if (data.error) {
      errorMsg.textContent = data.error;
      errorMsg.classList.add('show');
    } else {
      renderResult(data);
    }
  } catch (err) {
    errorMsg.textContent = 'Network error — is the server running?';
    errorMsg.classList.add('show');
  } finally {
    spinner.classList.remove('active');
    scanBtn.disabled = false;
    scanBtn.textContent = 'Scan URL';
  }
});

function renderResult(d) {
  const verdict = document.getElementById('verdict');
  verdict.className = 'verdict ' + (d.is_phishing ? 'phish' : 'safe');

  document.getElementById('verdictIcon').textContent  = d.is_phishing ? '⚠️' : '✅';
  document.getElementById('verdictLabel').textContent  = d.prediction;
  document.getElementById('verdictConf').textContent   = `${d.confidence}% confidence`;

  const phishPct = d.probabilities.phishing;
  const riskFill = document.getElementById('riskFill');
  riskFill.style.width = phishPct + '%';
  riskFill.style.background = phishPct > 70
    ? '#e74c3c'
    : phishPct > 40
      ? '#f39c12'
      : '#2ecc71';
  document.getElementById('riskLabel').textContent = d.risk;

  document.getElementById('safeProb').textContent  = d.probabilities.safe + '%';
  document.getElementById('phishProb').textContent = d.probabilities.phishing + '%';

  const tbody = document.getElementById('featureBody');
  tbody.innerHTML = '';
  for (const f of d.top_features) {
    const tr = document.createElement('tr');
    const name = f.name.replace(/_/g, ' ');
    tr.innerHTML = `<td>${name}</td><td>${f.value}</td><td>${f.importance}%</td>`;
    tbody.appendChild(tr);
  }

  result.classList.add('show');
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
