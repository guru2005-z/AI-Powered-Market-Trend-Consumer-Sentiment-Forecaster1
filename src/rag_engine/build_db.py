import chromadb
from chromadb.utils import embedding_functions
import json
import os
import glob

# --- CONFIGURATION ---
INPUT_DIR = "data/processed"
DB_PATH = "data/chromadb"  # Matches your folder structure
COLLECTION_NAME = "boat_market_insights"

def build_database():
    print(f"--- 🏗️ Building Vector Database in {DB_PATH} ---")

    # 1. Connect to ChromaDB (Persistent means it saves to disk)
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 2. Setup Embedding Function (Converts text -> numbers)
    print("   Loading Embedding Model (all-MiniLM-L6-v2)...")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # 3. Create or Reset Collection
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("   Deleted old collection to start fresh.")
    except:
        pass
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn
    )

    # 4. Find all chunk files (YouTube, News, Reddit)
    chunk_files = glob.glob(os.path.join(INPUT_DIR, "*_ready_chunks.json"))
    
    if not chunk_files:
        print("❌ No chunk files found! Run src/processing/chunker.py first.")
        return

    # 5. Process each file
    total_docs = 0
    for file_path in chunk_files:
        filename = os.path.basename(file_path)
        print(f"   Processing: {filename}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            
        documents = []
        metadatas = []
        ids = []
        
        for item in chunks:
            documents.append(item['text'])
            ids.append(item['chunk_id'])
            
            # Chroma requires flat metadata (no nested dicts, no nulls)
            meta = item['metadata']
            clean_meta = {k: str(v) if v is not None else "" for k, v in meta.items()}
            metadatas.append(clean_meta)
            
        # Add to DB in batches of 100
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            collection.add(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
        total_docs += len(documents)
            
    print(f"\n✅ Success! Database populated with {total_docs} memories.")
    print(f"   Saved to: {DB_PATH}")

if __name__ == "__main__":
    build_database()