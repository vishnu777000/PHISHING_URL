import os
import json
import pickle
import platform
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, matthews_corrcoef,
                             confusion_matrix, classification_report, roc_curve, precision_recall_curve,
                             ConfusionMatrixDisplay)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_DIR, "validation.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    print("=== STEP 1: Load Data ===")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    print(f"Train shape: {train_df.shape}, Label distribution:\n{train_df['label'].value_counts(normalize=True)}")
    print(f"Validation shape: {val_df.shape}, Label distribution:\n{val_df['label'].value_counts(normalize=True)}")
    print(f"Test shape: {test_df.shape}, Label distribution:\n{test_df['label'].value_counts(normalize=True)}")

    print("\n=== STEP 2: Define Feature Sets ===")
    all_features = [col for col in train_df.columns if col != 'label']
    no_https_features = [f for f in all_features if f != 'has_https']
    print(f"All features count: {len(all_features)}")
    print(f"No-HTTPS features count: {len(no_https_features)}")
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    print("\n=== STEP 3: Define Models ===")
    models = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
        ]),
        'DecisionTree': DecisionTreeClassifier(max_depth=15, min_samples_split=10, min_samples_leaf=5, class_weight='balanced', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=10, min_samples_leaf=5, max_features='sqrt', class_weight='balanced', n_jobs=-1, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, min_samples_split=10, min_samples_leaf=5, subsample=0.8, random_state=42)
    }

    feature_sets = {
        'all_features': all_features,
        'no_https': no_https_features
    }

    results = []
    trained_models = {}

    print("\n=== STEP 3 & 4: Train and Evaluate All Models on VALIDATION Set ===")
    for model_name, model in models.items():
        for fs_name, features in feature_sets.items():
            config_name = f"{model_name}_{fs_name}"
            print(f"Training {config_name}...")
            
            X_train = train_df[features].values
            X_val = val_df[features].values
            
            clf = sklearn.base.clone(model)
            clf.fit(X_train, y_train)
            trained_models[config_name] = {'model': clf, 'features': features, 'feature_names': features, 'model_name': model_name, 'fs_name': fs_name}
            
            y_pred = clf.predict(X_val)
            y_prob = clf.predict_proba(X_val)[:, 1]
            
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            roc = roc_auc_score(y_val, y_prob)
            pr = average_precision_score(y_val, y_prob)
            mcc = matthews_corrcoef(y_val, y_pred)
            
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            combined_score = 0.3 * f1 + 0.3 * mcc + 0.2 * roc + 0.2 * (1 - fpr)
            
            results.append({
                'config': config_name,
                'model': model_name,
                'feature_set': fs_name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1': f1,
                'ROC_AUC': roc,
                'PR_AUC': pr,
                'MCC': mcc,
                'FPR': fpr,
                'FNR': fnr,
                'Score': combined_score
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='Score', ascending=False)
    results_df.to_csv(os.path.join(BASE_DIR, "model_comparison.csv"), index=False)
    print("\nModel Comparison (Validation Set):")
    print(results_df[['config', 'F1', 'ROC_AUC', 'MCC', 'FPR', 'Score']])

    print("\n=== STEP 5: Select Best Model ===")
    best_config_name = results_df.iloc[0]['config']
    print(f"Selected Model: {best_config_name} with Score: {results_df.iloc[0]['Score']:.4f}")
    
    best_info = trained_models[best_config_name]
    best_model = best_info['model']
    best_features = best_info['features']
    
    X_val_best = val_df[best_features].values
    y_prob_best_val = best_model.predict_proba(X_val_best)[:, 1]

    print("\n=== STEP 6: Threshold Calibration on Validation Set ===")
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    thresh_results = []
    
    for th in thresholds:
        y_pred_th = (y_prob_best_val >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred_th).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        prec = precision_score(y_val, y_pred_th, zero_division=0)
        rec = recall_score(y_val, y_pred_th, zero_division=0)
        f1 = f1_score(y_val, y_pred_th, zero_division=0)
        
        thresh_results.append({
            'threshold': th,
            'FPR': fpr,
            'FNR': fnr,
            'Precision': prec,
            'Recall': rec,
            'F1': f1
        })
    
    thresh_df = pd.DataFrame(thresh_results)
    print("\nThreshold Calibration Table:")
    print(thresh_df)
    
    valid_thresholds = thresh_df[thresh_df['FPR'] < 0.05]
    if not valid_thresholds.empty:
        selected_row = valid_thresholds.loc[valid_thresholds['F1'].idxmax()]
    else:
        fallback = thresh_df[thresh_df['F1'] > 0.90]
        if not fallback.empty:
            selected_row = fallback.loc[fallback['FPR'].idxmin()]
        else:
            selected_row = thresh_df.loc[thresh_df['F1'].idxmax()]
            
    selected_threshold = selected_row['threshold']
    print(f"\nSelected Threshold: {selected_threshold}")
    
    high_risk_threshold = 0.80
    suspicious_threshold = max(0.20, selected_threshold * 0.6)

    print("\n=== STEP 7: Final Evaluation on TEST Set ===")
    X_test_best = test_df[best_features].values
    y_prob_test = best_model.predict_proba(X_test_best)[:, 1]
    y_pred_test = (y_prob_test >= selected_threshold).astype(int)
    
    test_acc = accuracy_score(y_test, y_pred_test)
    test_prec = precision_score(y_test, y_pred_test, zero_division=0)
    test_rec = recall_score(y_test, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    test_roc = roc_auc_score(y_test, y_prob_test)
    test_pr = average_precision_score(y_test, y_prob_test)
    test_mcc = matthews_corrcoef(y_test, y_pred_test)
    
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_pred_test).ravel()
    test_fpr = fp_test / (fp_test + tn_test) if (fp_test + tn_test) > 0 else 0
    test_fnr = fn_test / (fn_test + tp_test) if (fn_test + tp_test) > 0 else 0

    print("Test Metrics:")
    print(f"Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}")
    print(f"ROC-AUC: {test_roc:.4f}, PR-AUC: {test_pr:.4f}, MCC: {test_mcc:.4f}")
    print(f"FPR: {test_fpr:.4f}, FNR: {test_fnr:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_test))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, zero_division=0))

    print("\n=== STEP 8: Generate Plots ===")
    cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
    plt.title('Confusion Matrix (Test Set)')
    plt.savefig(os.path.join(ANALYSIS_DIR, 'confusion_matrix.png'))
    plt.close()

    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_prob_test)
    plt.figure()
    plt.plot(fpr_curve, tpr_curve, label=f'ROC curve (AUC = {test_roc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test Set)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(ANALYSIS_DIR, 'roc_curve.png'))
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, y_prob_test)
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AP = {test_pr:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Test Set)')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(ANALYSIS_DIR, 'precision_recall_curve.png'))
    plt.close()

    print("\n=== STEP 9: Save Feature Importance ===")
    if hasattr(best_model, 'feature_importances_') or (isinstance(best_model, Pipeline) and hasattr(best_model.named_steps.get('classifier'), 'feature_importances_')):
        if isinstance(best_model, Pipeline):
            importances = best_model.named_steps['classifier'].feature_importances_
        else:
            importances = best_model.feature_importances_
            
        feat_imp_df = pd.DataFrame({'feature': best_features, 'importance': importances})
        feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=False)
        feat_imp_df.to_csv(os.path.join(BASE_DIR, 'feature_importance.csv'), index=False)
        
        top_n = min(20, len(best_features))
        top_features = feat_imp_df.head(top_n)
        plt.figure(figsize=(10, 8))
        plt.barh(top_features['feature'][::-1], top_features['importance'][::-1])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, 'feature_importance.png'))
        plt.close()
    elif isinstance(best_model, Pipeline) and hasattr(best_model.named_steps.get('classifier'), 'coef_'):
        coefs = np.abs(best_model.named_steps['classifier'].coef_[0])
        feat_imp_df = pd.DataFrame({'feature': best_features, 'importance': coefs})
        feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=False)
        feat_imp_df.to_csv(os.path.join(BASE_DIR, 'feature_importance.csv'), index=False)
        
        top_n = min(20, len(best_features))
        top_features = feat_imp_df.head(top_n)
        plt.figure(figsize=(10, 8))
        plt.barh(top_features['feature'][::-1], top_features['importance'][::-1])
        plt.xlabel('Absolute Coefficient (Importance)')
        plt.title(f'Top {top_n} Feature Importances (LR)')
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, 'feature_importance.png'))
        plt.close()

    print("\n=== STEP 10: Save Model and Metadata ===")
    with open(os.path.join(MODELS_DIR, 'phishing_model_final.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
        
    with open(os.path.join(MODELS_DIR, 'feature_names_final.pkl'), 'wb') as f:
        pickle.dump(best_features, f)
        
    threshold_data = {
        "threshold": float(selected_threshold),
        "high_risk_threshold": float(high_risk_threshold),
        "suspicious_threshold": float(suspicious_threshold),
        "selected_on": "validation",
        "validation_f1_at_threshold": float(selected_row['F1']),
        "validation_fpr_at_threshold": float(selected_row['FPR'])
    }
    
    with open(os.path.join(MODELS_DIR, 'threshold.json'), 'w') as f:
        json.dump(threshold_data, f, indent=4)
        
    model_metadata = {
        "model_type": best_model.__class__.__name__ if not isinstance(best_model, Pipeline) else best_model.named_steps['classifier'].__class__.__name__,
        "feature_set": best_info['fs_name'],
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "feature_count": len(best_features),
        "feature_names": best_features,
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "training_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "threshold": float(selected_threshold),
        "test_metrics": {
            "accuracy": float(test_acc),
            "precision": float(test_prec),
            "recall": float(test_rec),
            "f1": float(test_f1),
            "roc_auc": float(test_roc),
            "pr_auc": float(test_pr),
            "mcc": float(test_mcc),
            "fpr": float(test_fpr),
            "fnr": float(test_fnr)
        }
    }
    
    with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=4)
        
    print("Pipeline execution complete. All artifacts saved.")

if __name__ == '__main__':
    main()