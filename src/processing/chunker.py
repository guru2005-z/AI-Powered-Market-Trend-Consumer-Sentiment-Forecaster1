import pandas as pd
import json
import os

# --- CONFIGURATION ---
INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/processed"
CHUNK_SIZE = 500  # Characters per chunk (good for reviews)
CHUNK_OVERLAP = 50 # Overlap to keep context between cuts

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Splits long text into smaller overlapping chunks.
    """
    if not isinstance(text, str) or len(text) < 1:
        return []
    
    # If text is short, return it as a single chunk
    if len(text) <= size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        # Try to find the last space within the limit to avoid cutting words
        if end < len(text):
            # Look for space in the last 10% of the chunk
            last_space = text.rfind(' ', start, end)
            if last_space != -1 and last_space > start + (size * 0.8):
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move forward, minus the overlap
        start = end - overlap
        # Prevent infinite loops if overlap >= step
        if start >= end:
            start = end
            
    return chunks

def process_file(filename):
    input_path = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(input_path):
        print(f"Skipping {filename}: File not found.")
        return

    print(f"Chunking {filename}...")
    
    # Auto-detect file type
    if filename.endswith('.csv'):
        df = pd.read_csv(input_path)
    elif filename.endswith('.json'):
        df = pd.read_json(input_path)
    else:
        return

    # Identify the text column (clean_text or content)
    text_col = 'clean_text' if 'clean_text' in df.columns else 'content'
    if text_col not in df.columns:
        print(f"  Skipping: No text column found in {filename}")
        return

    all_chunks = []

    for _, row in df.iterrows():
        original_text = row.get(text_col, "")
        text_chunks = chunk_text(original_text)
        
        # Create a "Vector Document" for each chunk
        for chunk in text_chunks:
            record = {
                "chunk_id": f"{filename}_{len(all_chunks)}",
                "text": chunk, # This is what gets embedded
                "metadata": {
                    "source": filename,
                    "original_date": str(row.get('date', '') or row.get('published_at', '')),
                    "rating": float(row.get('rating', 0)),
                    "sentiment_label": row.get('sentiment', 'Unknown') # If you ran sentiment analysis already
                }
            }
            all_chunks.append(record)

    # Save as JSON (Vector DBs love JSON)
    output_filename = filename.replace('.csv', '_chunks.json').replace('_cleaned', '_ready')
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)
    
    print(f"  ✅ Created {len(all_chunks)} chunks -> {output_path}")

def main():
    # Scan processed directory for cleaned files
    if not os.path.exists(INPUT_DIR):
        print("Processed data directory not found.")
        return

    files = [f for f in os.listdir(INPUT_DIR) if '_cleaned.csv' in f]
    
    if not files:
        print("No cleaned files found. Run cleaner.py first.")
        return

    for f in files:
        process_file(f)

if __name__ == "__main__":
    main()