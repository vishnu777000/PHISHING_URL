import sys
import os
from pathlib import Path
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / 'training'))

try:
    from feature_extraction import extract_features
except ImportError:
    print("Error: Could not import feature_extraction. Make sure feature_extraction.py exists in the training directory.")
    sys.exit(1)

CATEGORIES = {
    "A: Indian Universities": [
        'https://www.iitb.ac.in',
        'https://www.iitb.ac.in/en/education/academic-divisions',
        'https://www.iitb.ac.in/newacadhome/login.jsp',
        'https://portal.iitb.ac.in/asc/Login',
        'https://www.iitd.ac.in',
        'https://www.iitd.ac.in/academics',
        'https://www.iitd.ac.in/admission',
        'https://www.iitm.ac.in',
        'https://www.iitm.ac.in/students',
        'https://www.iitm.ac.in/happenings/exam-schedule',
        'https://www.du.ac.in',
        'https://www.du.ac.in/admission2024',
        'https://exam.du.ac.in/results',
        'https://www.jnu.ac.in',
        'https://www.jnu.ac.in/admission',
        'https://www.vit.ac.in',
        'https://www.vit.ac.in/admissions/btech',
        'https://vtop.vit.ac.in/vtop/login',
        'https://www.bits-pilani.ac.in',
        'https://www.bits-pilani.ac.in/admissions',
        'https://www.mu.ac.in',
        'https://www.mu.ac.in/results',
        'https://www.ignou.ac.in',
        'https://www.ignou.ac.in/ignou/studentzone',
        'https://ignou.samarth.edu.in/index.php/site/login',
        'https://www.nitt.edu',
        'https://www.nitt.edu/home/students/',
        'https://www.annauniv.edu',
        'https://www.annauniv.edu/coe1/results.php',
        'https://www.iisc.ac.in',
    ],
    "B: International Universities": [
        'https://www.mit.edu',
        'https://www.mit.edu/education',
        'https://student.mit.edu/cgi-bin/sfprwtrm.sh',
        'https://www.stanford.edu',
        'https://www.stanford.edu/admission',
        'https://login.stanford.edu',
        'https://www.harvard.edu',
        'https://www.harvard.edu/admissions-aid',
        'https://my.harvard.edu',
        'https://www.ox.ac.uk',
        'https://www.ox.ac.uk/admissions',
        'https://www.cam.ac.uk',
        'https://www.cam.ac.uk/admissions',
        'https://www.yale.edu',
        'https://www.yale.edu/admissions',
        'https://www.princeton.edu',
        'https://www.columbia.edu',
        'https://www.berkeley.edu',
        'https://www.ucla.edu',
        'https://www.utoronto.ca',
    ],
    "C: Government Educational Sites": [
        'https://www.ugc.ac.in',
        'https://www.aicte-india.org',
        'https://www.cbse.gov.in',
        'https://www.cbse.gov.in/cbsenew/examination.html',
        'https://ncert.nic.in',
        'https://www.education.gov.in',
        'https://www.nta.ac.in',
        'https://www.nta.ac.in/ExamResult',
        'https://www.ed.gov',
        'https://www.education.gov.uk',
    ],
    "D: Banks": [
        'https://www.onlinesbi.sbi',
        'https://www.onlinesbi.sbi/personal/login',
        'https://netbanking.hdfcbank.com',
        'https://www.icicibank.com',
        'https://infinity.icicibank.com/corp/Login',
        'https://www.axisbank.com',
        'https://www.chase.com',
        'https://secure.chase.com/web/auth/login',
        'https://www.bankofamerica.com',
        'https://www.bankofamerica.com/online-banking/sign-in',
        'https://www.wellsfargo.com',
        'https://connect.secure.wellsfargo.com/auth/login',
        'https://online.citi.com/US/login.do',
        'https://www.pnb.com.ph',
        'https://www.barclays.co.uk',
    ],
    "E: Major Tech Companies": [
        'https://www.google.com',
        'https://accounts.google.com/signin',
        'https://mail.google.com/mail/u/0/',
        'https://www.microsoft.com',
        'https://login.microsoftonline.com',
        'https://portal.azure.com',
        'https://www.apple.com',
        'https://support.apple.com',
        'https://www.amazon.com',
        'https://www.amazon.in/ap/signin',
        'https://www.github.com',
        'https://github.com/login',
        'https://www.linkedin.com',
        'https://www.linkedin.com/login',
        'https://www.facebook.com',
    ],
    "F: Government Websites": [
        'https://www.india.gov.in',
        'https://www.digilocker.gov.in',
        'https://www.incometax.gov.in',
        'https://www.incometax.gov.in/iec/foportal/',
        'https://parivahan.gov.in',
        'https://www.usa.gov',
        'https://www.gov.uk',
        'https://www.whitehouse.gov',
        'https://www.irs.gov',
        'https://www.service.gov.uk/sign-in',
    ],
    "G: News Websites": [
        'https://www.thehindu.com',
        'https://www.ndtv.com',
        'https://timesofindia.indiatimes.com',
        'https://www.nytimes.com',
        'https://www.bbc.com/news',
        'https://www.reuters.com',
        'https://www.theguardian.com',
        'https://www.washingtonpost.com',
        'https://www.cnn.com',
        'https://www.aljazeera.com',
    ],
    "H: E-commerce": [
        'https://www.amazon.in',
        'https://www.flipkart.com',
        'https://www.flipkart.com/account/login',
        'https://www.myntra.com',
        'https://www.walmart.com',
        'https://www.ebay.com',
        'https://www.ebay.com/signin',
        'https://www.shopify.com',
        'https://www.etsy.com',
        'https://www.alibaba.com',
    ],
    "I: Login Portals": [
        'https://login.gov',
        'https://accounts.google.com',
        'https://login.yahoo.com',
        'https://login.microsoftonline.com',
        'https://id.apple.com',
        'https://signin.aws.amazon.com/console',
        'https://login.salesforce.com',
        'https://www.dropbox.com/login',
        'https://zoom.us/signin',
        'https://slack.com/signin',
        'https://www.paypal.com/signin',
        'https://auth.atlassian.com',
        'https://sso.godaddy.com',
        'https://login.live.com',
        'https://discord.com/login',
    ],
    "J: Long URLs / Query Parameters": [
        'https://www.google.com/search?q=phishing+detection+machine+learning&hl=en&source=hp',
        'https://scholar.google.com/scholar?q=url+classification&as_sdt=0%2C5&as_ylo=2020',
        'https://www.amazon.in/dp/B08N5WRWNW/ref=cm_sw_r_cp_api_i_dl_XXXX',
        'https://en.wikipedia.org/wiki/Phishing#Types_of_phishing',
        'https://stackoverflow.com/questions/12345678/how-to-detect-phishing-urls',
        'https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf',
        'https://docs.google.com/document/d/1BxZ4RWjFpmGxFn5E2_P0JTq-YK9TmKN0HMfqZQbMDjk/edit',
        'https://mail.google.com/mail/u/0/?tab=rm&ogbl#inbox',
        'https://calendar.google.com/calendar/u/0/r/month/2024/9',
        'https://outlook.office365.com/mail/inbox/id/AAMkAGI2TG93AAA%3D',
    ]
}

PHISHING_URLS = [
    'http://paypal-login-verify.tk/signin',
    'http://secure-bankofamerica.ml/login',
    'http://apple-id-verify.cf/account',
    'http://microsoft-support.gq/security',
    'http://www.netflix-payment.xyz/billing',
    'http://192.168.1.100/paypal/login.html',
    'http://10.0.0.1:8080/bank/verify',
    'https://login-facebook-verify.web.app',
    'https://secure-wellsfargo.firebaseapp.com',
    'http://amazon-order-confirm.buzz/account',
    'http://google-security-alert.click/verify',
    'http://instagram-support.link/login',
    'https://appleid-recovery.pantheonsite.io',
    'http://linkedin-verify.rest/signin',
    'http://chase-secure-login.cam/auth',
    'http://paypal.com.verify-account.tk/login',
    'http://www.amaz0n-update.top/account',
    'http://micros0ft-alert.club/security',
    'http://www.g00gle-verify.icu/account',
    'http://faceb00k-security.work/login',
    'http://bit.ly/3abc123',
    'http://tinyurl.com/suspicious-link',
    'https://random-string-8f3k2.web.app/index.html',
    'http://xn--pypal-4ve.com/login',
    'http://www.update-your-bank-account.com/verify',
    'http://secure-login-portal.info/auth/signin',
    'http://account-recovery-center.xyz/reset',
    'http://verify-your-identity.top/confirm',
    'http://payment-failed-notification.buzz/update',
    'http://suspicious-domain.gq/banking/login.php?ref=email&id=12345',
]

def load_model():
    model_path = BASE_DIR / 'models' / 'phishing_model_final.pkl'
    if not model_path.exists():
        model_path = BASE_DIR / 'model' / 'phishing_model.pkl'
        
    feature_path = BASE_DIR / 'models' / 'feature_names_final.pkl'
    if not feature_path.exists():
        feature_path = BASE_DIR / 'model' / 'feature_names.pkl'

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
        
    if not feature_path.exists():
        print(f"Error: Feature names file not found at {feature_path}")
        sys.exit(1)

    threshold_path = BASE_DIR / 'models' / 'threshold.json'
    threshold = 0.50
    if threshold_path.exists():
        with open(threshold_path) as f:
            threshold = json.load(f).get('threshold', 0.50)
            
    model = joblib.load(model_path)
    feature_names = joblib.load(feature_path)
    
    return model, feature_names, threshold

def analyze_prediction(url, model, feature_names, threshold, is_phishing_test=False):
    features = extract_features(url)
    features_df = pd.DataFrame([features])
    features_df = features_df.reindex(columns=feature_names, fill_value=0)
    
    proba = model.predict_proba(features_df)[0][1]
    is_phishing = proba >= threshold
    
    top_features_str = ""
    is_incorrect = (not is_phishing_test and is_phishing) or (is_phishing_test and not is_phishing)
    
    if is_incorrect and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_vals = features_df.iloc[0]
        contributions = np.abs(feature_vals) * importances
        top_indices = np.argsort(contributions)[::-1][:5]
        
        top_features = []
        for idx in top_indices:
            feat_name = feature_names[idx]
            val = feature_vals[feat_name]
            if contributions[idx] > 0:
                top_features.append(f"{feat_name}: {val:.2f}")
                
        if top_features:
            top_features_str = " | Top features: " + ", ".join(top_features)

    return proba, is_phishing, top_features_str

def main():
    print("Loading model...")
    model, feature_names, threshold = load_model()
    
    print("\n======================================================================")
    print("  FALSE-POSITIVE & PHISHING RECALL TEST")
    print("======================================================================")
    
    total_legit = 0
    correct_legit = 0
    college_legit = 0
    college_correct = 0
    
    for category_name, urls in CATEGORIES.items():
        print(f"\n--- CATEGORY {category_name} ---")
        cat_total = len(urls)
        cat_correct = 0
        
        for url in urls:
            proba, is_phishing, top_feats = analyze_prediction(url, model, feature_names, threshold, is_phishing_test=False)
            total_legit += 1
            if "A: Indian" in category_name or "B: International" in category_name:
                college_legit += 1
            
            if not is_phishing:
                cat_correct += 1
                correct_legit += 1
                if "A: Indian" in category_name or "B: International" in category_name:
                    college_correct += 1
                print(f"  [PASS] BENIGN ({100-proba*100:.1f}%) {url}")
            else:
                print(f"  [FAIL] PHISHING ({proba*100:.1f}%) {url}  <-- FALSE POSITIVE{top_feats}")
                
        print(f"\n  Category {category_name.split(':')[0]} Results: {cat_correct}/{cat_total} correct ({(cat_correct/cat_total)*100:.1f}%)")
        print(f"  False positives: {cat_total - cat_correct}")
        
    print("\n======================================================================")
    print("  LEGITIMATE URL SUMMARY")
    print("======================================================================")
    
    fpr = ((total_legit - correct_legit) / total_legit) * 100 if total_legit > 0 else 0
    college_fpr = ((college_legit - college_correct) / college_legit) * 100 if college_legit > 0 else 0
    
    print(f"\n  Total legitimate URLs tested: {total_legit}")
    print(f"  Correctly classified BENIGN: {correct_legit}")
    print(f"  FALSE POSITIVES (legitimate flagged as phishing): {total_legit - correct_legit}")
    print(f"  False positive rate: {fpr:.1f}%\n")
    
    print(f"  College/University subset:")
    print(f"    College URLs tested: {college_legit}")
    print(f"    Correctly classified BENIGN: {college_correct}")
    print(f"    Incorrectly classified PHISHING: {college_legit - college_correct}")
    print(f"    College false positive rate: {college_fpr:.1f}%")

    print("\n======================================================================")
    print("  PHISHING RECALL TEST")
    print("======================================================================")
    print("")
    
    total_phishing = len(PHISHING_URLS)
    correct_phishing = 0
    
    for url in PHISHING_URLS:
        proba, is_phishing, top_feats = analyze_prediction(url, model, feature_names, threshold, is_phishing_test=True)
        
        if is_phishing:
            correct_phishing += 1
            print(f"  [PASS] PHISHING ({proba*100:.1f}%) {url}")
        else:
            print(f"  [FAIL] BENIGN ({100-proba*100:.1f}%) {url}  <-- FALSE NEGATIVE{top_feats}")
            
    fnr = ((total_phishing - correct_phishing) / total_phishing) * 100 if total_phishing > 0 else 0
    recall = (correct_phishing / total_phishing) * 100 if total_phishing > 0 else 0
            
    print(f"\n  Phishing URLs tested: {total_phishing}")
    print(f"  Correctly detected PHISHING: {correct_phishing}")
    print(f"  Missed (false negatives): {total_phishing - correct_phishing}")
    print(f"  Phishing recall: {recall:.1f}%")
    print(f"  False negative rate: {fnr:.1f}%")

    print("\n======================================================================")
    print("  OVERALL SUMMARY")
    print("======================================================================")
    print(f"\n  Legitimate URLs: {total_legit} tested, {correct_legit} correct, FPR = {fpr:.1f}%")
    print(f"  Phishing URLs: {total_phishing} tested, {correct_phishing} correct, FNR = {fnr:.1f}%\n")
    print(f"  College URLs: {college_legit} tested, {college_correct} correct, College FPR = {college_fpr:.1f}%\n")
    
    both_low = "YES" if (fpr < 10 and fnr < 10) else "NO"
    print(f"  BOTH low false positives AND low false negatives: [{both_low}]")
    print("======================================================================")

if __name__ == '__main__':
    main()
