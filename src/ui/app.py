import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import yagmail
import re
from datetime import date
from dotenv import load_dotenv

# --- 1. SETUP PAGE CONFIG (Must be first) ---
st.set_page_config(page_title="AI Market Forecaster", layout="wide")

# --- 2. LAZY IMPORTS (Prevents Crashing on Startup) ---
# We only import heavy AI libraries inside functions, not at the top.
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load Environment Variables
load_dotenv()

# --- 3. HELPER FUNCTIONS ---
@st.cache_data
def load_data():
    try:
        # Create dummy data if file missing (Prevents FileNotFoundError crash)
        if not os.path.exists("data/processed/youtube_sentiment.csv"):
            return pd.DataFrame({'content': ['No data'], 'sentiment': [0], 'date': [pd.Timestamp.now()]}), 'date'
            
        df = pd.read_csv("data/processed/youtube_sentiment.csv")
        # Date parsing logic...
        date_col = None
        for col in ['date', 'published_at', 'timestamp', 'created_at']:
            if col in df.columns:
                date_col = col
                break
        if date_col is None:
            df['date'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df))
            date_col = 'date'
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        return df.sort_values(by=date_col), date_col
    except Exception as e:
        return pd.DataFrame(), None

# --- 4. AI PIPELINE (Cached & Lazy Loaded) ---
@st.cache_resource
def get_ai_engine():
    """Initializes the heavy AI models only when needed."""
    # Check Keys
    pinecone_key = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
    google_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not pinecone_key or not google_key:
        return None, None

    # Import here to save RAM on startup
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from langchain.chains import RetrievalQA

    # Load Model (This is the heavy part)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = PineconeVectorStore(
        index_name="market-forecaster",
        embedding=embeddings,
        pinecone_api_key=pinecone_key
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=google_key
    )
    
    return vectorstore, llm

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Overview", "📈 Trend Analytics", "🔔 Alerts & Reports"])
    st.divider()
    
    # 🚀 BUILD DATABASE BUTTON
    if st.button("🚀 Build Database (Remote)"):
        with st.spinner("⏳ Building Database... (This takes ~60s)"):
            try:
                import build_db  # This runs your build_db.py script
                build_db.build_db()
                st.success("✅ Database Built Successfully!")
                time.sleep(2)
                st.rerun()
            except ImportError:
                st.error("❌ Error: 'build_db.py' file not found.")
            except Exception as e:
                st.error(f"❌ Build Failed: {e}")

    # 💬 CHATBOT
    st.header("💬 AI Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Ask me about the market trends!"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Run AI only now
        with st.spinner("Thinking..."):
            v_store, llm_model = get_ai_engine()
            if v_store and llm_model:
                from langchain.chains import RetrievalQA
                qa = RetrievalQA.from_chain_type(
                    llm=llm_model, chain_type="stuff", 
                    retriever=v_store.as_retriever()
                )
                response = qa.invoke({"query": prompt})
                ans = response["result"]
            else:
                ans = "⚠️ Please set your API Keys in Settings -> Secrets."
            
            st.session_state.messages.append({"role": "assistant", "content": ans})
            st.chat_message("assistant").write(ans)

# --- 6. MAIN CONTENT ---
df, date_col = load_data()

if page == "Overview":
    st.title("📊 Market Overview")
    if not df.empty:
        st.metric("Total Reviews", len(df))
        # Simple Charts
        daily = df.groupby(pd.Grouper(key=date_col, freq='D')).size().reset_index(name='counts')
        st.line_chart(daily, x=date_col, y='counts')
    else:
        st.info("No data found. Click 'Build Database' in the sidebar!")

elif page == "📈 Trend Analytics":
    st.title("📈 Trends")
    st.write("Analytics View")

elif page == "🔔 Alerts & Reports":
    st.title("🔔 Alerts")
    st.write("Alerts View")