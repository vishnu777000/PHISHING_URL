import pandas as pd
import tldextract
from sklearn.model_selection import GroupShuffleSplit
import os

def get_registrable_domain(hostname):
    if pd.isna(hostname):
        return ""
    ext = tldextract.extract(str(hostname))
    if ext.registered_domain:
        return ext.registered_domain
    return str(hostname)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(base_dir, 'dataset', 'features.csv')
    
    df = pd.read_csv(features_path)
    
    df['group'] = df['domain'].apply(get_registrable_domain)
    
    unique_domains = df['group'].nunique()
    
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    trainval_idx, test_idx = next(gss1.split(df, groups=df['group']))
    
    trainval_df = df.iloc[trainval_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, val_idx = next(gss2.split(trainval_df, groups=trainval_df['group']))
    
    train_df = trainval_df.iloc[train_idx].copy()
    val_df = trainval_df.iloc[val_idx].copy()
    
    train_domains = set(train_df['group'])
    val_domains = set(val_df['group'])
    test_domains = set(test_df['group'])
    
    train_val_overlap = len(train_domains.intersection(val_domains))
    train_test_overlap = len(train_domains.intersection(test_domains))
    val_test_overlap = len(val_domains.intersection(test_domains))
    
    assert train_val_overlap == 0, "Domain overlap between train and validation"
    assert train_test_overlap == 0, "Domain overlap between train and test"
    assert val_test_overlap == 0, "Domain overlap between validation and test"
    
    print("DOMAIN-AWARE DATASET SPLIT")
    print("==========================")
    print(f"\nDataset: {len(df)} rows, {len(df.columns) - 4} features")
    print(f"Unique registrable domains: {unique_domains}")
    
    total = len(df)
    print(f"\nSplit sizes:")
    print(f"  Train:      {len(train_df)} rows ({len(train_df)/total*100:.1f}%)")
    print(f"  Validation: {len(val_df)} rows ({len(val_df)/total*100:.1f}%)")
    print(f"  Test:       {len(test_df)} rows ({len(test_df)/total*100:.1f}%)")
    
    print("\nLabel distribution:")
    train_labels = train_df['label'].value_counts()
    val_labels = val_df['label'].value_counts()
    test_labels = test_df['label'].value_counts()
    
    def format_labels(vc):
        benign = vc.get(0, 0)
        phish = vc.get(1, 0)
        return f"Benign={benign}, Phishing={phish}"
    
    print(f"  Train:      {format_labels(train_labels)}")
    print(f"  Validation: {format_labels(val_labels)}")
    print(f"  Test:       {format_labels(test_labels)}")
    
    print("\nDomain overlap verification:")
    print(f"  TRAIN vs VALIDATION = {train_val_overlap} ({'PASS' if train_val_overlap == 0 else 'FAIL'})")
    print(f"  TRAIN vs TEST = {train_test_overlap} ({'PASS' if train_test_overlap == 0 else 'FAIL'})")
    print(f"  VALIDATION vs TEST = {val_test_overlap} ({'PASS' if val_test_overlap == 0 else 'FAIL'})")
    
    print(f"\nTrain domains: {len(train_domains)}")
    print(f"Validation domains: {len(val_domains)}")
    print(f"Test domains: {len(test_domains)}")
    
    cols_to_drop = ['url', 'domain', 'group']
    train_df = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
    val_df = val_df.drop(columns=[c for c in cols_to_drop if c in val_df.columns])
    test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])
    
    train_path = os.path.join(base_dir, 'dataset', 'train.csv')
    val_path = os.path.join(base_dir, 'dataset', 'validation.csv')
    test_path = os.path.join(base_dir, 'dataset', 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

if __name__ == '__main__':
    main()