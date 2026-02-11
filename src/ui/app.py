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

# --- 1. SETUP PAGE CONFIG ---
st.set_page_config(page_title="AI Market Forecaster", layout="wide")

# --- 2. UNIVERSAL IMPORTS (Works on ALL versions) ---
try:
    # Try New Style (v0.3+)
    from langchain_pinecone import PineconeVectorStore
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.chains import RetrievalQA
    from langchain_core.prompts import PromptTemplate
except ImportError:
    # Try Old Style (v0.1 - v0.2)
    from langchain_community.vectorstores import Pinecone as PineconeVectorStore
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

# Load Keys
load_dotenv()

# --- 3. HELPER FUNCTIONS ---
@st.cache_data
def load_data():
    try:
        if not os.path.exists("data/processed/youtube_sentiment.csv"):
            return pd.DataFrame(), None
        df = pd.read_csv("data/processed/youtube_sentiment.csv")
        # Simple date fix
        df['date'] = pd.to_datetime(df.get('date', pd.Timestamp.now()), errors='coerce')
        return df, 'date'
    except Exception:
        return pd.DataFrame(), None

# --- 4. AI PIPELINE ---
@st.cache_resource
def setup_ai():
    # Get Keys
    pinecone_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    
    if not pinecone_key or not google_key:
        return None, None

    # Load AI
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

# --- 5. UI & SIDEBAR ---
st.title("🤖 AI Market Forecaster")

with st.sidebar:
    st.header("⚙️ Actions")
    
    # DATABASE BUILDER
    if st.button("🚀 Build Database"):
        with st.spinner("Building..."):
            try:
                import build_db
                build_db.build_db()
                st.success("✅ Done!")
            except ImportError:
                st.error("❌ 'build_db.py' not found.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    st.divider()
    
    # CHATBOT
    st.header("💬 Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Run AI
        v_store, llm_model = setup_ai()
        if v_store and llm_model:
            qa = RetrievalQA.from_chain_type(
                llm=llm_model, chain_type="stuff", 
                retriever=v_store.as_retriever()
            )
            try:
                response = qa.invoke({"query": prompt})
                ans = response["result"]
            except:
                # Fallback for older LangChain
                ans = qa.run(prompt)
            
            st.session_state.messages.append({"role": "assistant", "content": ans})
            st.chat_message("assistant").write(ans)
        else:
            st.error("⚠️ Please set API Keys in Secrets.")

# --- 6. MAIN CONTENT ---
df, date_col = load_data()
if not df.empty:
    st.subheader("Overview")
    st.metric("Total Reviews", len(df))
    st.line_chart(df['sentiment'])
else:
    st.info("👋 Welcome! Click 'Build Database' to start.")