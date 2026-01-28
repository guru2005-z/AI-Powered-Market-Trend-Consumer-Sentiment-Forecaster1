import os
import sys
import json
from chromadb.config import Settings
import chromadb
from dotenv import load_dotenv

# Load API Keys
load_dotenv()

# --- CONFIGURATION ---
# This MUST match the path used in build_db.py
DB_PATH = "data/chromadb"
COLLECTION_NAME = "boat_market_insights"

def get_rag_chain():
    """
    Creates a simple retriever that connects to ChromaDB.
    Returns a tuple of (client, collection) that can be used to query the database.
    """
    
    # 1. Check Database
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database not found at {DB_PATH}")
        print("   Run 'python src/rag_engine/build_db.py' first.")
        return None, None

    # 2. Connect to ChromaDB
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        print(f"✅ Connected to ChromaDB at {DB_PATH}")
        
        # Get all collections to see what's available
        collections = client.list_collections()
        print(f"✅ Available collections: {[c.name for c in collections]}")
        
        # Try to get the collection
        if any(c.name == COLLECTION_NAME for c in collections):
            collection = client.get_collection(name=COLLECTION_NAME)
            print(f"✅ Loaded collection: {COLLECTION_NAME}")
            return client, collection
        else:
            print(f"⚠️  Collection '{COLLECTION_NAME}' not found")
            return client, None
    except Exception as e:
        print(f"❌ Error connecting to ChromaDB: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# --- TEST BLOCK ---
# This runs only if you execute this file directly
if __name__ == "__main__":
    print("--- 🤖 Testing ChromaDB Retriever ---\n")
    
    client, collection = get_rag_chain()
    
    if collection is not None:
        # Test Query
        query = "What are the common complaints about battery life?"
        print(f"\n❓ Searching for: {query}")
        print("⏳ Searching...")
        
        try:
            # Query the collection with a simple text search first
            results = collection.query(
                query_texts=[query],
                n_results=3
            )
            
            if results and results['documents'] and len(results['documents'][0]) > 0:
                print(f"\n✅ Found {len(results['documents'][0])} results:\n")
                for i, doc in enumerate(results['documents'][0]):
                    print(f"   [{i+1}] {doc[:120]}...")
                    
                if results['metadatas'] and results['metadatas'][0]:
                    print("\n🔍 Metadata:")
                    for i, meta in enumerate(results['metadatas'][0]):
                        source = meta.get('source', 'Unknown') if isinstance(meta, dict) else 'Unknown'
                        print(f"   [{i+1}] Source: {source}")
            else:
                print("\n✅ Collection is ready, but empty. Run build_db.py to populate it.")
                
        except Exception as e:
            print(f"❌ Error during query: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Could not initialize retriever.")