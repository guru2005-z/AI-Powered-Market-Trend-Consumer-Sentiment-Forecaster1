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
    clean_str = re.sub(r'[^\d.]', '', str(price_str)) 
    try:
        return float(clean_str)
    except ValueError:
        return 0.0

def clean_text(text):
    """
    Aggressive cleaning: Removes everything except text content.
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = re.sub(r'http\S+|https\S+|www\.\S+|ftp\S+|ftps\S+', '', text)
    text = re.sub(r'[\w\-]+\.[\w\-]+/\S+', '', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[#@]\w+', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[\\\/|]', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_rating(rating):
    if pd.isna(rating):
        return 0.0
    match = re.search(r'(\d+(\.\d+)?)', str(rating))
    if match:
        return float(match.group(1))
    return 0.0


def ensure_dir(path):
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

    # --- FIX START: Include date_col in desired_columns ---
    desired_columns = ['title', 'content', 'text']
    
    # If a date column is defined and exists in the DF, make sure we keep it
    if date_col and date_col in df.columns:
        desired_columns.append(date_col)
    
    available_columns = [col for col in desired_columns if col in df.columns]
    
    if available_columns:
        df = df[available_columns]
    else:
        # If none of the desired columns exist, keep first 3 columns as fallback
        df = df.iloc[:, :3]
    # --- FIX END ---

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
    
    # Convert date to string for JSON serialization compatibility
    if date_col and date_col in df.columns:
         df[date_col] = df[date_col].astype(str)

    df.to_csv(out_csv, index=False)
    df.to_json(out_json, orient='records', indent=4)

    print("------------------------------------------------")
    print(f"Saved cleaned dataset: {out_csv}")
    print(f"Saved cleaned dataset: {out_json}")
    print(f"Rows: {len(df)}")


def process_pipeline():
    input_files = []

    if os.path.exists(CSV_FILE_PATH):
        input_files.append(CSV_FILE_PATH)
    if os.path.exists(JSON_FILE_PATH):
        input_files.append(JSON_FILE_PATH)

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

        # YouTube (comments) configuration
        if 'youtube' in fname:
            # Note: mapping 'date' or 'publishedAt' to 'published_at'
            mapping = {'date': 'published_at', 'publishedAt': 'published_at', 'title': 'title', 'text': 'content'}
            text_cols = ['title', 'content']
            dedupe = ['content'] 
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
                'Review Count': 'review_count',
                'date': 'date' # Ensure date is mapped if present in news
            }
            text_cols = ['product_title', 'description', 'content']
            dedupe = ['product_title']
            fill = {'price': 0.0, 'rating': 0.0, 'review_count': 0, 'description': 'No description available'}
            
            # Check if 'date' exists after mapping, otherwise set None
            date_col = 'date' 
            basename = os.path.splitext(os.path.basename(fpath))[0]

        process_and_export(df, basename, mapping=mapping, text_columns=text_cols, dedupe_subset=dedupe, fill_values=fill, date_col=date_col)

    print("All files processed.")

if __name__ == "__main__":
    process_pipeline()