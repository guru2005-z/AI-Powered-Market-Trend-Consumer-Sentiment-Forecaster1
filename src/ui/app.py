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

# --- 2. EXACT IMPORTS (DO NOT CHANGE) ---
# These match your requirements.txt perfectly.
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load Keys
load_dotenv()

# --- 3. HELPER FUNCTIONS (DATA LOADING) ---
@st.cache_data
def load_data():
    try:
        if not os.path.exists("data/processed/youtube_sentiment.csv"):
            return pd.DataFrame(), None
        df = pd.read_csv("data/processed/youtube_sentiment.csv")
        
        # Date Logic
        date_col = None
        for col in ['date', 'published_at', 'timestamp', 'created_at']:
            if col in df.columns:
                date_col = col
                break
        if date_col is None:
            df['date'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df))
            date_col = 'date'
        
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(by=date_col)
        return df, date_col
    except Exception:
        return pd.DataFrame(), None

@st.cache_data
def load_topics():
    try:
        if not os.path.exists("data/processed/topics_summary.json"):
            return {}
        with open("data/processed/topics_summary.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# --- 4. FEATURE FUNCTIONS (CHARTS, ALERTS, EMAIL) ---

def process_topics_for_chart(json_data):
    if not json_data or "topics" not in json_data:
        return pd.DataFrame()
    results = []
    for item in json_data["topics"]:
        feature_name = item.get("name", "Unknown")
        pos_count = len(item.get("positive", []))
        neg_count = len(item.get("negative", []))
        total = pos_count + neg_count + len(item.get("neutral", []))
        if total > 0:
            score = (pos_count - neg_count) / total
            results.append({"Feature": feature_name, "Sentiment Score": score, "Mentions": total})
    return pd.DataFrame(results)

def check_alerts(df, threshold=-0.2):
    alerts = []
    if df.empty: return alerts
    recent_days = df.tail(3)
    avg_recent = recent_days['sentiment'].mean()
    if avg_recent < threshold:
        alerts.append(f"🚨 CRITICAL: Sentiment dropped to {avg_recent:.2f} (Threshold: {threshold})")
    neg_reviews = len(recent_days[recent_days['sentiment'] == -1])
    if neg_reviews > 5:
        alerts.append(f"⚠️ WARNING: High volume of negative reviews detected ({neg_reviews} in 3 days).")
    return alerts

def format_markdown_to_html(text):
    # Simple converter for emails
    text = re.sub(r'^# (.*)', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*)', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = text.replace("\n", "<br>")
    return text

def send_email_to_boss(to_email, subject, content):
    user_email = os.getenv("GMAIL_USER") or st.secrets.get("GMAIL_USER")
    app_password = os.getenv("GMAIL_APP_PASSWORD") or st.secrets.get("GMAIL_APP_PASSWORD")

    if not user_email or not app_password:
        return False, "❌ Error: Missing GMAIL secrets."

    formatted_html = format_markdown_to_html(content[0])
    html_body = f"""
    <div style="font-family: Arial; padding: 20px;">
        <h2 style="color: #003366;">Executive Market Report</h2>
        <div style="background: #f9f9f9; padding: 15px;">{formatted_html}</div>
    </div>
    """
    try:
        yag = yagmail.SMTP(user=user_email, password=app_password)
        yag.send(to=to_email, subject=subject, contents=[html_body])
        return True, "✅ Sent!"
    except Exception as e:
        return False, f"❌ Failed: {e}"

# --- 5. AI PIPELINE ---
@st.cache_resource
def setup_ai():
    # Get Keys
    pinecone_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    
    if not pinecone_key or not google_key:
        return None, None

    try:
        # Load AI
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name="market-forecaster",
            embedding=embeddings
        )
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=google_key
        )
        return vectorstore, llm
    except Exception:
        return None, None

# --- 6. UI & SIDEBAR ---
st.title("🤖 AI Market Forecaster")

# Load Data Once
df, date_col = load_data()
topics_json = load_topics()
topic_text = json.dumps(topics_json)[:1000] if topics_json else "No topic data."

# Sidebar Navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Overview", "📈 Trend Analytics", "🔔 Alerts & Reports"])
    st.divider()
    
    # DATABASE BUILDER
    if st.button("🚀 Build Database"):
        with st.spinner("Building..."):
            try:
                import build_db
                build_db.build_db()
                st.success("✅ Database Built! Reloading...")
                st.rerun()
            except ImportError:
                st.error("❌ 'build_db.py' not found.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # CHATBOT
    st.divider()
    st.header("💬 Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about the market..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Run AI
        v_store, llm_model = setup_ai()
        if v_store and llm_model:
            # Setup QA Chain
            template = f"""Answer based on context. 
            Context: {{context}}
            Market Data: {topic_text}
            Question: {{question}}
            Answer:"""
            
            PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
            
            qa = RetrievalQA.from_chain_type(
                llm=llm_model, chain_type="stuff", 
                retriever=v_store.as_retriever(),
                chain_type_kwargs={"prompt": PROMPT}
            )
            ans = qa.run(prompt)
            
            st.session_state.messages.append({"role": "assistant", "content": ans})
            st.chat_message("assistant").write(ans)
        else:
            st.error("⚠️ AI not ready. Check API Keys.")

# --- 7. MAIN PAGES ---

if page == "Overview":
    st.subheader("📊 Market Overview")
    if not df.empty:
        total = len(df)
        pos = len(df[df['sentiment'] == 1])
        neg = len(df[df['sentiment'] == -1])
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Reviews", total)
        c2.metric("Positive", pos)
        c3.metric("Negative", neg)
        
        # Charts
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Sentiment Split")
            fig_pie = px.pie(names=["Positive", "Negative", "Neutral"], 
                             values=[pos, neg, total-pos-neg],
                             color_discrete_sequence=["#2ecc71", "#e74c3c", "#95a5a6"])
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            st.markdown("### Topic Scores")
            f_df = process_topics_for_chart(topics_json)
            if not f_df.empty:
                fig_bar = px.bar(f_df, x="Feature", y="Sentiment Score", color="Sentiment Score")
                st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No data. Click 'Build Database' to start.")

elif page == "📈 Trend Analytics":
    st.subheader("📈 Trends")
    if not df.empty and date_col:
        daily = df.groupby(pd.Grouper(key=date_col, freq='D')).agg(
            Avg=('sentiment', 'mean'), Vol=('content', 'count')).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily[date_col], y=daily['Vol'], name='Volume', marker_color='silver'))
        fig.add_trace(go.Scatter(x=daily[date_col], y=daily['Avg'], name='Sentiment', line=dict(color='blue')))
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔔 Alerts & Reports":
    st.subheader("🔔 Alerts & Reports")
    
    # Alerts
    alerts = check_alerts(df)
    if alerts:
        for a in alerts: st.error(a)
    else:
        st.success("✅ System Healthy")
        
    st.divider()
    
    # Report Gen
    if st.button("Generate AI Report"):
        v_store, llm = setup_ai()
        if llm:
            with st.spinner("Writing..."):
                prompt = f"Write a professional market report based on this summary: {topic_text}"
                rep = llm.invoke(prompt).content
                st.session_state.rep = rep
    
    if "rep" in st.session_state:
        st.markdown(st.session_state.rep)
        
        # Email Form
        with st.form("email"):
            email = st.text_input("Boss Email")
            if st.form_submit_button("Send"):
                ok, msg = send_email_to_boss(email, "Market Report", [st.session_state.rep])
                if ok: st.success(msg)
                else: st.error(msg)