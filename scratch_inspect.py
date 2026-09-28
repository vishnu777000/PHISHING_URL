"""Quick inspection script for dataset analysis."""
import pandas as pd

df = pd.read_csv('dataset/phising_urls.csv')
benign = df[df['label']=='benign']
phish = df[df['label']=='phishing']

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"Total URLs: {len(df)}")
print(f"Benign: {len(benign)}")
print(f"Phishing: {len(phish)}")
print(f"Ratio (Benign:Phishing): {len(benign)/len(phish):.2f}:1")

print("\n" + "=" * 70)
print("INSTITUTIONAL URL ANALYSIS")
print("=" * 70)

edu_patterns = ['.edu', '.ac.in', '.edu.in', '.ac.uk', 'university', 'college', 'institute', 'school']
for pat in edu_patterns:
    b_count = benign['URL'].str.lower().str.contains(pat, regex=False).sum()
    p_count = phish['URL'].str.lower().str.contains(pat, regex=False).sum()
    print(f'  "{pat}" -> Benign: {b_count}, Phishing: {p_count}')

print("\n" + "=" * 70)
print("KEYWORD ANALYSIS (Benign vs Phishing)")
print("=" * 70)

keywords = ['login', 'signin', 'student', 'faculty', 'admission', 'exam', 'result',
            'library', 'portal', 'account', 'verify', 'payment', 'secure', 'bank']
for pat in keywords:
    b_count = benign['URL'].str.lower().str.contains(pat, regex=False).sum()
    p_count = phish['URL'].str.lower().str.contains(pat, regex=False).sum()
    print(f'  "{pat}" -> Benign: {b_count}, Phishing: {p_count}')

print("\n" + "=" * 70)
print("HTTPS DISTRIBUTION")
print("=" * 70)

print(f'Benign HTTPS: {benign["URL"].str.startswith("https").sum()} / {len(benign)} ({benign["URL"].str.startswith("https").mean()*100:.1f}%)')
print(f'Benign HTTP:  {benign["URL"].str.startswith("http://").sum()} / {len(benign)} ({benign["URL"].str.startswith("http://").mean()*100:.1f}%)')
print(f'Phishing HTTPS: {phish["URL"].str.startswith("https").sum()} / {len(phish)} ({phish["URL"].str.startswith("https").mean()*100:.1f}%)')
print(f'Phishing HTTP:  {phish["URL"].str.startswith("http://").sum()} / {len(phish)} ({phish["URL"].str.startswith("http://").mean()*100:.1f}%)')

print("\n" + "=" * 70)
print("URL STRUCTURE ANALYSIS")
print("=" * 70)

benign_with_path = benign['URL'].str.count('/').gt(3).sum()
benign_with_query = benign['URL'].str.contains('?', regex=False).sum()
phish_with_path = phish['URL'].str.count('/').gt(3).sum()
phish_with_query = phish['URL'].str.contains('?', regex=False).sum()
print(f'Benign with deep paths (>3 slashes): {benign_with_path} / {len(benign)}')
print(f'Phishing with deep paths (>3 slashes): {phish_with_path} / {len(phish)}')
print(f'Benign with query params: {benign_with_query} / {len(benign)}')
print(f'Phishing with query params: {phish_with_query} / {len(phish)}')

print(f'\nBenign URL length - mean: {benign["URL"].str.len().mean():.1f}, median: {benign["URL"].str.len().median():.1f}, max: {benign["URL"].str.len().max()}')
print(f'Phishing URL length - mean: {phish["URL"].str.len().mean():.1f}, median: {phish["URL"].str.len().median():.1f}, max: {phish["URL"].str.len().max()}')

print("\n" + "=" * 70)
print("SAMPLE .edu BENIGN URLs")
print("=" * 70)
edu_benign = benign[benign['URL'].str.lower().str.contains('.edu', regex=False)]
for url in edu_benign['URL'].head(20):
    print(f'  {url}')

print("\n" + "=" * 70)
print("SAMPLE BENIGN LOGIN URLs")
print("=" * 70)
login_benign = benign[benign['URL'].str.lower().str.contains('login', regex=False)]
for url in login_benign['URL'].head(20):
    print(f'  {url}')
