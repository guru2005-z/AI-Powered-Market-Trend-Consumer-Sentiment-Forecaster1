import os
import time
import pandas as pd
from dotenv import load_dotenv

# --- MODERN IMPORTS ---
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

# --- CONFIGURATION ---
INPUT_FILE = "data/processed/youtube_sentiment.csv"
INDEX_NAME = "market-forecaster"
DESIRED_DIMENSION = 384  # Correct dimension for all-MiniLM-L6-v2

def build_vector_db():
    print("--- 🏗️ Building Vector Database ---")
    
    # 1. Initialize Pinecone Client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # 2. Check and Fix Index Dimensions
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    if INDEX_NAME in existing_indexes:
        index_info = pc.describe_index(INDEX_NAME)
        current_dim = int(index_info.dimension)
        
        if current_dim != DESIRED_DIMENSION:
            print(f"⚠️ Mismatch detected! Index '{INDEX_NAME}' has {current_dim} dims, but we need {DESIRED_DIMENSION}.")
            print(f"♻️ Deleting incompatible index '{INDEX_NAME}'...")
            pc.delete_index(INDEX_NAME)
            time.sleep(10) # Wait for deletion to propagate
            existing_indexes.remove(INDEX_NAME)
        else:
            print(f"✅ Index '{INDEX_NAME}' already exists with correct dimensions.")

    # 3. Create Index if it doesn't exist
    if INDEX_NAME not in existing_indexes:
        print(f"🆕 Creating new index: '{INDEX_NAME}' with dimension {DESIRED_DIMENSION}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DESIRED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        # Wait for index to be ready
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
        print("✅ Index is ready!")

    # 4. Setup Embedding Model
    print("Loading Embedding Model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 5. Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"❌ File not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    # Ensure content exists
    df = df.dropna(subset=['content'])
    print(f"📂 Loaded {len(df)} reviews.")

    # 6. Convert to Documents
    documents = []
    print("Converting rows to vector documents...")
    
    for _, row in df.iterrows():
        doc = Document(
            page_content=str(row['content']),
            metadata={"sentiment": str(row.get('sentiment', '0'))}
        )
        documents.append(doc)

    # 7. Upload
    print(f"🚀 Uploading {len(documents)} vectors to Pinecone...")
    PineconeVectorStore.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("✅ Success! Database Built.")

if __name__ == "__main__":
    build_vector_db()