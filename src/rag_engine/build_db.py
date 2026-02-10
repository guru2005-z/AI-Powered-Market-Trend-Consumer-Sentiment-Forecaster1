import os
import time
import pandas as pd
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
# ✅ OFFLINE MODEL (Bypasses your WiFi Block)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

# --- CONFIGURATION ---
INDEX_NAME = "market-forecaster"
CSV_FILE_PATH = "data/processed/youtube_sentiment.csv"
DESIRED_DIMENSION = 384  # <--- Correct size for Offline Model

def build_db():
    print("--- 🏗️ STARTING OFFLINE DATABASE BUILD ---")
    
    # 1. API Key Check
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        print("❌ Error: PINECONE_API_KEY missing in .env")
        return

    # 2. Load Offline Model
    print("💻 Loading Local AI Brain (all-MiniLM-L6-v2)...")
    try:
        # This downloads the model once (~80MB) and runs on your CPU
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("✅ Local AI Loaded Successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("👉 Run: pip install sentence-transformers langchain-huggingface")
        return

    # 3. Connect to Pinecone & Auto-Fix Index
    print("🌲 Connecting to Pinecone...")
    pc = Pinecone(api_key=pinecone_key)
    
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    # A. Check existing index
    if INDEX_NAME in existing_indexes:
        print(f"🔍 Checking existing index '{INDEX_NAME}'...")
        index_info = pc.describe_index(INDEX_NAME)
        
        # If dimensions are wrong (e.g. 768 from your screenshot), DELETE IT.
        if int(index_info.dimension) != DESIRED_DIMENSION:
            print(f"♻️ WRONG DIMENSIONS DETECTED (Found {index_info.dimension}, Need {DESIRED_DIMENSION}).")
            print("🗑️ Deleting incompatible index (This takes 20s)...")
            pc.delete_index(INDEX_NAME)
            time.sleep(20)
            existing_indexes = [i.name for i in pc.list_indexes()]
    
    # B. Create Index automatically
    if INDEX_NAME not in existing_indexes:
        print(f"🆕 Creating NEW Index '{INDEX_NAME}' (384 Dim)...")
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=DESIRED_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for readiness
            while not pc.describe_index(INDEX_NAME).status['ready']:
                time.sleep(1)
            print("✅ Index Created Successfully!")
        except Exception as e:
            print(f"❌ Index Creation Error: {e}")
            return

    # 4. Load Data
    if not os.path.exists(CSV_FILE_PATH):
        print(f"❌ Error: File not found at {CSV_FILE_PATH}")
        return
    
    df = pd.read_csv(CSV_FILE_PATH).dropna(subset=['content'])
    print(f"📂 Loaded {len(df)} reviews.")

    # 5. Prepare Documents
    documents = []
    for _, row in df.iterrows():
        doc = Document(
            page_content=str(row['content'])[:1000],
            metadata={"sentiment": str(row.get('sentiment', '0'))}
        )
        documents.append(doc)

    # 6. Upload (Fast & Local)
    print("🚀 Uploading to Pinecone...")
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=pinecone_key
    )

    batch_size = 100 
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        try:
            print(f"   🔹 Uploading batch {i} to {i+len(batch)}...", end="\r")
            vectorstore.add_documents(batch)
            print(f"   ✅ Batch {i} OK!       ")
        except Exception as e:
            print(f"   ⚠️ Batch error: {e}")

    print("\n✅ SUCCESS! Database Built (Offline Mode).")

if __name__ == "__main__":
    build_db()