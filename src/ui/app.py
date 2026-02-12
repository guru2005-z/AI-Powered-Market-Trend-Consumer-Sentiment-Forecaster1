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

# --- 2. STABLE IMPORTS (Matches your Requirements.txt) ---
try:
    from langchain_community.vectorstores import Pinecone as PineconeVectorStore
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    st.error("⚠️ Library Version Mismatch. Please check requirements.txt")
    st.stop()

# Load Keys
load_dotenv()

# --- 3. HELPER FUNCTIONS (Data & Charts) ---
@st.cache_data
def load_data():
    try:
        if not os.path.exists("data/processed/youtube_sentiment.csv"):
            return pd.DataFrame(), None
        df = pd.read_csv("data/processed/youtube_sentiment.csv")
        
        # Smart Date Parsing
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
    except Exception:
        return pd.DataFrame(), None

@st.cache_data
def load_topics():
    try:
        with open("data/processed/topics_summary.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def process_topics_for_chart(json_data):
    if not json_data or "topics" not in json_data:
        return pd.DataFrame()
    results = []
    for item in json_data["topics"]:
        feature_name = item.get("name", "Unknown")
        pos = len(item.get("positive", []))
        neg = len(item.get("negative", []))
        total = pos + neg + len(item.get("neutral", []))
        if total > 0:
            score = (pos - neg) / total
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

# --- 4. EMAIL HELPERS ---
def format_markdown_to_html(text):
    text = re.sub(r'^# (.*)', r'<h2 style="color:#003366; border-bottom: 2px solid #eee;">\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*)', r'<h3 style="color:#00509E;">\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'^\* (.*)', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = text.replace("\n", "<br>")
    return text

def send_email_to_boss(to_email, subject, content):
    user_email = os.getenv("GMAIL_USER") or st.secrets.get("GMAIL_USER")
    app_password = os.getenv("GMAIL_APP_PASSWORD") or st.secrets.get("GMAIL_APP_PASSWORD")

    if not user_email or not app_password:
        return False, "❌ Missing GMAIL credentials in Secrets."

    formatted_html = format_markdown_to_html(content)
    try:
        yag = yagmail.SMTP(user=user_email, password=app_password)
        yag.send(to=to_email, subject=subject, contents=[formatted_html])
        return True, "✅ Report sent successfully!"
    except Exception as e:
        return False, f"❌ Email failed: {e}"

# --- 5. AI ENGINE (Stable Setup) ---
@st.cache_resource
def setup_rag_chain():
    # 1. Get Keys
    pinecone_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    
    if not pinecone_key or not google_key:
        return None, None

    # 2. Initialize Models
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name="market-forecaster",
            embedding=embeddings
        )
    except:
        return None, None # Index might not exist yet

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=google_key
    )

    # 3. Create Chain
    template = """
    You are a Market Analyst AI. Answer based on the context below.
    
    CONTEXT:
    {context}
    
    QUESTION: {question}
    
    ANSWER:
    """
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain, llm

# --- 6. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("Navigation")
    
    # 🚀 DATABASE BUILDER (From New Code)
    if st.button("🚀 Build Database"):
        with st.spinner("Building..."):
            try:
                import build_db
                build_db.build_db()
                st.success("✅ Done! Reloading...")
                st.rerun()
            except ImportError:
                st.error("❌ 'build_db.py' not found.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    st.divider()
    page = st.radio("Go to", ["Overview", "📈 Trend Analytics", "🔔 Alerts & Reports"])
    st.divider()

    # 💬 CHATBOT (From Old Code)
    st.header("💬 AI Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Ask me about market trends!"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        qa_chain, _ = setup_rag_chain()
        if qa_chain:
            with st.spinner("Thinking..."):
                try:
                    res = qa_chain.run(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": res})
                    st.chat_message("assistant").write(res)
                except Exception as e:
                    st.error(f"AI Error: {e}")
        else:
            st.error("⚠️ Database not ready. Click 'Build Database' first.")

# --- 7. MAIN PAGES ---
df, date_col = load_data()
topics_json = load_topics()
qa_chain, llm_direct = setup_rag_chain()

if page == "Overview":
    st.title("📊 Market Overview")
    if not df.empty:
        total = len(df)
        pos = len(df[df['sentiment'] == 1])
        neg = len(df[df['sentiment'] == -1])
        neu = len(df[df['sentiment'] == 0])
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Reviews", total)
        c2.metric("Positive", pos, f"{((pos/total)*100):.1f}%")
        c3.metric("Negative", neg, f"{((neg/total)*100):.1f}%", delta_color="inverse")
        
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Sentiment Distribution")
            fig = px.pie(names=["Positive", "Negative", "Neutral"], values=[pos, neg, neu], 
                         color_discrete_sequence=["#2ecc71", "#e74c3c", "#95a5a6"])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Feature Scores")
            f_df = process_topics_for_chart(topics_json)
            if not f_df.empty:
                fig = px.bar(f_df, x="Feature", y="Sentiment Score", color="Sentiment Score", range_y=[-1,1])
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👋 Welcome! Click '🚀 Build Database' in the sidebar to load data.")

elif page == "📈 Trend Analytics":
    st.title("📈 Trends")
    if not df.empty and date_col:
        daily = df.groupby(pd.Grouper(key=date_col, freq='D')).agg(
            Avg=('sentiment', 'mean'), Vol=('content', 'count')).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily[date_col], y=daily['Vol'], name='Volume', marker_color='silver', yaxis='y2'))
        fig.add_trace(go.Scatter(x=daily[date_col], y=daily['Avg'], name='Sentiment', line=dict(color='blue', width=3)))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right'))
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔔 Alerts & Reports":
    st.title("🔔 Alerts & Reporting")
    
    # Alerts
    alerts = check_alerts(df)
    if alerts:
        for a in alerts: st.error(a)
    else:
        st.success("✅ No critical alerts.")
    
    st.divider()
    
    # Report Generation
    if st.button("Generate Executive Report"):
        if llm_direct:
            with st.spinner("Generating..."):
                prompt = f"Write a professional market report based on {len(df)} reviews with a sentiment score of {df['sentiment'].mean():.2f}."
                report = llm_direct.invoke(prompt).content
                st.session_state.report = report
    
    if "report" in st.session_state:
        st.markdown(st.session_state.report)
        
        # Email Form
        with st.form("email"):
            email = st.text_input("Recipient Email")
            if st.form_submit_button("Send Email"):
                success, msg = send_email_to_boss(email, "Market Report", st.session_state.report)
                if success: st.success(msg)
                else: st.error(msg)