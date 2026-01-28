import os
import pandas as pd
import re
import json

# --- CONFIGURATION ---
CSV_FILE_PATH = 'src/data/boat_news_raw.csv'
JSON_FILE_PATH = 'src/data/boat_news_raw.json'

# Directories
RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed' 

def clean_price(price_str):
    """
    Converts price strings like '₹2,999' or 'Rs. 2499' to float 2499.0
    """
    if pd.isna(price_str):
        return 0.0
    # Remove currency symbols, commas, and whitespace
    clean_str = re.sub(r'[^\d.]', '', str(price_str)) 
    try:
        return float(clean_str)
    except ValueError:
        return 0.0

def clean_text(text):
    """
    Aggressive cleaning: Removes everything except text content.
    Keeps only: Letters (a-z, A-Z), Numbers (0-9), Spaces, and basic punctuation (.,!?'-).
    Removes: URLs, links, emails, HTML, emojis, special characters, symbols, etc.
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # 1. Remove URLs (http, https, ftp, ftps, www)
    text = re.sub(r'http\S+|https\S+|www\.\S+|ftp\S+|ftps\S+', '', text)
    
    # 2. Remove shortened links and domains with paths
    text = re.sub(r'[\w\-]+\.[\w\-]+/\S+', '', text)
    
    # 3. Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # 4. Remove HTML and XML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # 5. Remove hashtags, mentions, and other social media symbols
    text = re.sub(r'[#@]\w+', '', text)
    
    # 6. Remove numbers followed by patterns like prices, codes, etc.
    # Keep numbers within text, but remove isolated hex codes and special number patterns
    
    # 7. Remove emojis and non-ASCII characters (keep only ASCII text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # 8. Remove backslashes, forward slashes, pipes, and other path separators
    text = re.sub(r'[\\\/|]', '', text)
    
    # 9. Keep ONLY: Letters, Numbers, Spaces, and basic punctuation (. , ! ? ' -)
    # Remove everything else (special symbols, brackets, etc.)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'\-]', '', text)
    
    # 10. Remove multiple spaces, tabs, newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_rating(rating):
    """
    Extracts the numeric rating from strings like '4.5 out of 5 stars'.
    """
    if pd.isna(rating):
        return 0.0
    match = re.search(r'(\d+(\.\d+)?)', str(rating))
    if match:
        return float(match.group(1))
    return 0.0


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def process_and_export(df, basename, mapping=None, text_columns=None, dedupe_subset=None, fill_values=None, date_col=None):
    """Standardized processing and export for a single dataset."""
    if mapping:
        df.rename(columns=mapping, inplace=True)

    # Text cleaning
    if text_columns:
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)

    # Numeric cleaning
    if 'price' in df.columns:
        df['price'] = df['price'].apply(clean_price)
    if 'mrp' in df.columns:
        df['mrp'] = df['mrp'].apply(clean_price)
    if 'rating' in df.columns:
        df['rating'] = df['rating'].apply(normalize_rating)

    # Date parsing
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Keep only text, content, and title columns
    desired_columns = ['title', 'content', 'text']
    available_columns = [col for col in desired_columns if col in df.columns]
    if available_columns:
        df = df[available_columns]
    else:
        # If none of the desired columns exist, keep first 3 columns
        df = df.iloc[:, :3]

    # Deduplicate using available columns
    if dedupe_subset:
        existing_subset = [c for c in dedupe_subset if c in df.columns]
        if existing_subset:
            df.drop_duplicates(subset=existing_subset, keep='first', inplace=True)

    # Fill missing values
    if fill_values:
        df.fillna(fill_values, inplace=True)

    # Export
    ensure_dir(PROCESSED_DIR)
    out_csv = os.path.join(PROCESSED_DIR, f"{basename}_cleaned.csv")
    out_json = os.path.join(PROCESSED_DIR, f"{basename}_cleaned.json")
    df.to_csv(out_csv, index=False)
    df.to_json(out_json, orient='records', indent=4)

    print("------------------------------------------------")
    print(f"Saved cleaned dataset: {out_csv}")
    print(f"Saved cleaned dataset: {out_json}")
    print(f"Rows: {len(df)}")


def process_pipeline():
    """Scan raw directories, detect dataset type, apply dataset-specific cleaning, and export cleaned files."""
    input_files = []

    # include legacy specific files if present
    if os.path.exists(CSV_FILE_PATH):
        input_files.append(CSV_FILE_PATH)
    if os.path.exists(JSON_FILE_PATH):
        input_files.append(JSON_FILE_PATH)

    # scan raw dir
    if os.path.isdir(RAW_DIR):
        for fname in os.listdir(RAW_DIR):
            if fname.lower().endswith(('.csv', '.json')):
                input_files.append(os.path.join(RAW_DIR, fname))

    if not input_files:
        print("No input files were found - nothing to process.")
        return

    for fpath in input_files:
        print(f"Processing file: {fpath}")
        try:
            if fpath.lower().endswith('.csv'):
                df = pd.read_csv(fpath)
            else:
                try:
                    df = pd.read_json(fpath)
                except ValueError:
                    df = pd.read_json(fpath, lines=True)
        except Exception as e:
            print(f"Failed to load {fpath}: {e}")
            continue

        fname = os.path.basename(fpath).lower()

        # YouTube (comments)
        if 'youtube' in fname:
            mapping = {'date': 'published_at', 'title': 'title', 'text': 'content'}
            text_cols = ['title', 'content']
            dedupe = ['url', 'content']
            fill = {'content': ''}
            date_col = 'published_at'
            basename = 'youtube'

        # Default / news / product data
        else:
            mapping = {
                'Product Name': 'product_title',
                'Title': 'product_title',
                'title': 'product_title',
                'Price': 'price',
                'MRP': 'mrp',
                'Rating': 'rating',
                'Description': 'description',
                'content': 'content',
                'Review Count': 'review_count'
            }
            text_cols = ['product_title', 'description', 'content']
            dedupe = ['product_title']
            fill = {'price': 0.0, 'rating': 0.0, 'review_count': 0, 'description': 'No description available'}
            date_col = None
            basename = os.path.splitext(os.path.basename(fpath))[0]

        # run standardized processing and export
        process_and_export(df, basename, mapping=mapping, text_columns=text_cols, dedupe_subset=dedupe, fill_values=fill, date_col=date_col)

    print("All files processed.")

if __name__ == "__main__":
    process_pipeline()