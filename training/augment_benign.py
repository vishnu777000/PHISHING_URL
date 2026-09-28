import os
import pandas as pd
from urllib.parse import urlparse

def get_domain(url):
    try:
        res = urlparse(url)
        netloc = res.netloc
        if not netloc:
            res = urlparse('http://' + url)
            netloc = res.netloc
        return netloc
    except:
        return ""

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    in_file = os.path.join(project_dir, 'dataset', 'cleaned_urls.csv')
    out_file = os.path.join(project_dir, 'dataset', 'augmented_urls.csv')
    
    try:
        df = pd.read_csv(in_file)
    except FileNotFoundError:
        print(f"Warning: {in_file} not found. Starting with empty dataframe.")
        df = pd.DataFrame(columns=['URL', 'label', 'Domain', 'length'])
        
    benign = df[df['label'] == 0]
    
    augmented_www = []
    for url in benign['URL']:
        if isinstance(url, str):
            if url.startswith('www.'):
                augmented_www.append(url.replace('www.', '', 1))
            elif url.startswith('http://www.'):
                augmented_www.append(url.replace('http://www.', 'http://', 1))
            elif url.startswith('https://www.'):
                augmented_www.append(url.replace('https://www.', 'https://', 1))
            elif not url.startswith('www.') and not url.startswith('http'):
                augmented_www.append('www.' + url)
                
    df_aug_www = pd.DataFrame({'URL': augmented_www, 'label': 0})
    
    categories = {
        'A_Indian_Universities': [
            "https://www.iitb.ac.in", "https://iitb.ac.in/en", "https://iitb.ac.in/student",
            "https://www.iitb.ac.in/admission", "https://iitb.ac.in/departments", "https://library.iitb.ac.in",
            "https://www.iitb.ac.in/placements",
            "https://iitd.ac.in", "https://home.iitd.ac.in/courses", "https://home.iitd.ac.in/faculty",
            "https://home.iitd.ac.in/research",
            "https://iitm.ac.in", "https://www.iitm.ac.in/students", "https://www.iitm.ac.in/academics",
            "https://du.ac.in", "https://admission.uod.ac.in", "https://du.ac.in/colleges", "https://exam.du.ac.in/results",
            "https://jnu.ac.in", "https://jnuee.jnu.ac.in/admission", "https://jnu.ac.in/academics",
            "https://vit.ac.in", "https://vit.ac.in/admissions", "https://vit.ac.in/exam", "https://vit.ac.in/results",
            "https://bits-pilani.ac.in", "https://bitsadmission.com",
            "https://nitt.edu", "https://www.nitt.edu/students",
            "https://annauniv.edu", "https://coe1.annauniv.edu/home/results",
            "https://mu.ac.in", "https://mu.ac.in/results",
            "https://ignou.ac.in", "https://ignou.samarth.edu.in/student/login", "https://ignou.ac.in/admission", "https://ignou.ac.in/results",
            "https://amity.edu", "https://amity.edu/admissions",
        ] + [f"https://www.{uni}.ac.in/{path}" for uni in ["bhu", "amu", "jmi", "uohyd", "pondiuni", "cusat", "manipal", "srmist", "sastra", "thapar"] for path in ["", "student/login", "exam/results", "faculty/profile", "library/catalog", "portal", "admission"]],
        
        'B_International_Universities': [
            "https://mit.edu", "https://mit.edu/admissions", "https://mit.edu/education", "https://mit.edu/research", "https://student.mit.edu/login",
            "https://stanford.edu", "https://stanford.edu/admissions", "https://stanford.edu/education", "https://axess.stanford.edu/login",
            "https://harvard.edu", "https://college.harvard.edu/admissions", "https://harvard.edu/academics", "https://key.harvard.edu/login",
            "https://ox.ac.uk", "https://ox.ac.uk/admissions", "https://ox.ac.uk/students", "https://bodleian.ox.ac.uk/libraries",
            "https://cam.ac.uk", "https://cam.ac.uk/admissions", "https://cam.ac.uk/students",
        ] + [f"https://www.{uni}.edu/{path}" for uni in ["yale", "princeton", "columbia", "upenn", "cornell", "brown", "dartmouth", "berkeley", "ucla", "ucsd", "ucsf"] for path in ["", "admissions/apply/graduate", "student/portal/dashboard", "faculty/directory/search", "login"]] +
        [f"https://www.{uni}.edu.au/{path}" for uni in ["unimelb", "sydney", "uq", "monash", "unsw"] for path in ["", "student/portal", "login"]] +
        [f"https://www.{uni}.ac.jp/en/{path}" for uni in ["u-tokyo", "kyoto-u", "osaka-u", "tohoku", "nagoya-u"] for path in ["", "admissions", "students"]] +
        [f"https://www.{uni}.ac.uk/{path}" for uni in ["imperial", "ucl", "kcl", "ed", "manchester"] for path in ["", "study/undergraduate", "student-life"]],
        
        'C_Legitimate_Logins': [
            "https://login.microsoftonline.com",
            "https://accounts.google.com/signin",
            "https://login.yahoo.com",
            "https://id.apple.com",
            "https://portal.office.com/account",
            "https://signin.aws.amazon.com",
            "https://login.salesforce.com",
            "https://onlinebanking.sbi.co.in/login",
            "https://netbanking.hdfcbank.com",
            "https://accounts.ecourts.gov.in",
            "https://www.incometax.gov.in/iec/foportal"
        ] + [f"https://{sub}.{uni}.edu/login" for uni in ["yale", "mit", "stanford", "harvard", "princeton"] for sub in ["sso", "cas", "login", "auth"]] +
        [f"https://netbanking.{bank}.com/login" for bank in ["icicibank", "axisbank", "kotak", "yesbank", "indusind"]],
        
        'D_Government_Ed': [
            "https://ugc.ac.in", "https://aicte-india.org", "https://cbse.gov.in",
            "https://ncert.nic.in", "https://education.gov.in", "https://ed.gov", "https://education.gov.uk"
        ] + [f"https://{state}educationboard.gov.in" for state in ["mh", "up", "mp", "ap", "tn", "ka", "rj", "gj", "wb", "pb"]],
        
        'E_Banks': [
            "https://www.onlinesbi.sbi", "https://netbanking.hdfcbank.com",
            "https://retail.onlinesbi.sbi/retail/login.htm",
            "https://netbanking.hdfcbank.com/netbanking/",
        ] + [f"https://www.{bank}.com/{path}" for bank in ["icicibank", "axisbank", "chase", "bankofamerica", "wellsfargo", "citi"] for path in ["", "login", "netbanking", "account/summary", "transactions", "personal/banking"]],
        
        'F_Tech_Companies': [
            "https://accounts.google.com", "https://drive.google.com", "https://mail.google.com/mail/u/0",
            "https://portal.azure.com", "https://login.live.com", "https://outlook.office365.com",
            "https://support.apple.com", "https://developer.apple.com",
            "https://aws.amazon.com/console", "https://seller.amazon.in",
        ] + [f"https://www.{tech}.com/{path}" for tech in ["facebook", "twitter", "linkedin", "github", "microsoft", "apple", "netflix"] for path in ["login", "account", "settings", "security", "verify", "auth/login", "users/settings"]],
        
        'G_Government': [
            "https://india.gov.in", "https://digilocker.gov.in", "https://parivahan.gov.in", "https://incometax.gov.in",
            "https://whitehouse.gov", "https://usa.gov", "https://irs.gov",
            "https://gov.uk", "https://nhs.uk",
        ] + [f"https://www.{dept}.gov.in" for dept in ["mha", "mea", "mof", "mohfw", "mod", "railways", "uidai", "epfindia", "passportindia", "rbi"]],
        
        'H_News': [
            "https://thehindu.com", "https://ndtv.com", "https://timesofindia.indiatimes.com",
            "https://nytimes.com", "https://bbc.com/news", "https://reuters.com"
        ] + [f"https://www.{news}.com/{path}" for news in ["cnn", "foxnews", "wsj", "washingtonpost", "theguardian", "bloomberg", "cnbc", "aljazeera"] for path in ["", "world", "politics", "business", "article/2024/09/20/latest-news", "search?query=election"]],
        
        'I_Ecommerce': [
            "https://amazon.in/dp/B09XYZ123", "https://flipkart.com/checkout/login",
            "https://walmart.com/account/login", "https://ebay.com/myaccount"
        ] + [f"https://www.{ecom}.com/{path}" for ecom in ["amazon", "flipkart", "myntra", "ajio", "meesho", "croma", "reliancedigital", "tatacliq"] for path in ["", "cart", "checkout/payment", "account/login", "orders/track", "product/detail?id=123"]],
        
        'J_Queries': [
            "https://university.edu/student/profile?id=12345",
            "https://results.university.edu/exam?sem=6&year=2024",
            "https://library.university.edu/catalog/search?q=machine+learning",
            "https://news.site.com/article?id=789&ref=homepage",
            "https://bank.com/account/statement?from=2024-01-01&to=2024-06-30"
        ] + [f"https://{site}.com/search?q={query}&sort={sort}" for site in ["amazon", "ebay", "flipkart", "youtube", "google"] for query in ["laptop", "phone", "shoes", "books", "watch"] for sort in ["relevance", "price_asc", "price_desc", "newest"]],
        
        'K_Subdomains': [
            "https://portal.university.edu", "https://results.university.edu",
            "https://mail.google.com", "https://drive.google.com",
            "https://developer.mozilla.org", "https://docs.github.com"
        ] + [f"https://{sub}.{domain}.com" for sub in ["mail", "docs", "dev", "api", "support", "help", "blog", "shop", "app", "dashboard"] for domain in ["google", "microsoft", "apple", "amazon", "github", "slack", "zoom", "salesforce"]]
    }
    
    new_urls = []
    print("--- Added URL Statistics ---")
    for cat, urls in categories.items():
        urls = list(set(urls))
        print(f"{cat}: {len(urls)} URLs")
        new_urls.extend(urls)
        
    print(f"\nTotal new institutional/legitimate URLs generated: {len(new_urls)}")
    
    df_new = pd.DataFrame({'URL': new_urls, 'label': 0})
    
    df_combined = pd.concat([df, df_aug_www, df_new], ignore_index=True)
    
    initial_len = len(df_combined)
    df_combined.drop_duplicates(subset=['URL'], inplace=True)
    print(f"Dropped {initial_len - len(df_combined)} duplicate URLs.")
    
    df_combined['Domain'] = df_combined['URL'].apply(get_domain)
    df_combined['length'] = df_combined['URL'].apply(lambda x: len(str(x)))
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    df_combined.to_csv(out_file, index=False)
    
    print(f"Saved augmented dataset to: {out_file}")
    print(f"Total rows in output: {len(df_combined)}")

if __name__ == "__main__":
    main()