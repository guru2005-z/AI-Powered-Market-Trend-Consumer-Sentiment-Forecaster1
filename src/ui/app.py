import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# --- STRICT MODERN IMPORTS ---
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
# 👇 NEW: Import Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# 1. Load Keys
load_dotenv()

# 2. Page Config
st.set_page_config(page_title="AI Market Forecaster", layout="wide")

# 3. Load Data Functions
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/processed/youtube_sentiment.csv")
        
        # --- DATA PRE-PROCESSING FOR TRENDS ---
        date_col = None
        for col in ['date', 'published_at', 'timestamp', 'created_at']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            dates = pd.date_range(end=pd.Timestamp.now(), periods=len(df))
            df['date'] = dates
            date_col = 'date'
        
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(by=date_col)
        
        return df, date_col

    except FileNotFoundError:
        return pd.DataFrame(), None

@st.cache_data
def load_topics():
    try:
        with open("data/processed/topics_summary.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# --- CHART PROCESSOR ---
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
            results.append({
                "Feature": feature_name,
                "Sentiment Score": score,
                "Mentions": total
            })

    return pd.DataFrame(results)

# 4. Setup AI Pipeline (Switched to Google Gemini)
@st.cache_resource
def setup_rag_pipeline_v4(topic_summary_text):
    # Check for Google Key
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        st.error("Missing API Keys! Make sure GOOGLE_API_KEY and PINECONE_API_KEY are in your .env file.")
        return None

    # 1. Embeddings (We switch to Google's Embeddings for better compatibility, 
    # OR we can keep using HuggingFace embeddings if you don't want to rebuild the DB.
    # PRO TIP: To avoid rebuilding the DB, we will stick with the original HF Embeddings for now.
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Connect to Pinecone
    vectorstore = PineconeVectorStore(
        index_name="market-forecaster",
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )
    
    # 3. Setup LLM -> GOOGLE GEMINI 🚀
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    template = """
    You are a market researcher. Use the specific TOPIC DATA below to guide your answer.
    
    --- KEY TOPIC DATA (Verified Facts) ---
    {topics}
    ---------------------------------------
    
    Reviews from Database:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
        partial_variables={"topics": topic_summary_text}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# --- INITIALIZATION ---
try:
    df, date_column = load_data()
    topics_json = load_topics()
    
    topic_text = json.dumps(topics_json, indent=2, ensure_ascii=False) if topics_json else "No topic data available."
    
    # Call the V4 pipeline (Gemini Version)
    qa_chain = setup_rag_pipeline_v4(topic_text)

except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "AI Dashboard", "📈 Trend Analytics"])

# --- PAGE 1: OVERVIEW ---
if page == "Overview":
    st.title("📊 Boat Market Overview")
    
    if not df.empty:
        total = len(df)
        positive = len(df[df['sentiment'] == 1])
        negative = len(df[df['sentiment'] == -1])
        neutral = len(df[df['sentiment'] == 0])
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Reviews", total)
        c2.metric("Positive", positive, delta=f"{((positive/total)*100):.1f}%")
        c3.metric("Negative", negative, delta_color="inverse", delta=f"{((negative/total)*100):.1f}%")
        
        st.divider()
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Overall Sentiment")
            fig_pie = px.pie(
                names=["Positive", "Negative", "Neutral"], 
                values=[positive, negative, neutral], 
                color_discrete_sequence=["#2ecc71", "#e74c3c", "#95a5a6"],
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_right:
            st.subheader("Feature Sentiment Score")
            feature_df = process_topics_for_chart(topics_json)
            if not feature_df.empty:
                fig_bar = px.bar(
                    feature_df, 
                    x="Feature", y="Sentiment Score",
                    color="Sentiment Score",
                    color_continuous_scale="RdYlGn",
                    range_y=[-1, 1],
                    text_auto=".2f"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No topic data found.")

# --- PAGE 2: AI DASHBOARD ---
elif page == "AI Dashboard":
    st.title("🤖 AI Analyst Dashboard (Powered by Gemini)")
    st.markdown("Ask deep questions. The AI uses **Topic Facts + Reviews**.")

    with st.sidebar:
        st.divider()
        st.header("🧠 Knowledge Base")
        if topics_json and "topics" in topics_json:
            st.success(f"Loaded {len(topics_json['topics'])} Topics")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about trend shifts, specific complaints..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Gemini is thinking..."):
                if qa_chain:
                    try:
                        response = qa_chain.invoke({"query": prompt})
                        answer = response["result"]
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.error("AI pipeline unavailable.")

# --- PAGE 3: TREND ANALYTICS ---
elif page == "📈 Trend Analytics":
    st.title("📈 Sentiment Trend Analysis")
    st.markdown("Track how consumer sentiment changes over time.")

    if df.empty or date_column is None:
        st.warning("No data or date column found to generate trends.")
    else:
        daily_trends = df.groupby(pd.Grouper(key=date_column, freq='D')).agg(
            Avg_Sentiment=('sentiment', 'mean'),
            Review_Volume=('content', 'count')
        ).reset_index()

        daily_trends['Smoothed_Sentiment'] = daily_trends['Avg_Sentiment'].rolling(window=7, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily_trends[date_column],
            y=daily_trends['Review_Volume'],
            name='Review Volume',
            marker_color='rgba(200, 200, 200, 0.5)',
            yaxis='y2'
        ))
        fig.add_trace(go.Scatter(
            x=daily_trends[date_column],
            y=daily_trends['Smoothed_Sentiment'],
            name='Sentiment Trend (7-Day Avg)',
            line=dict(color='#3498db', width=3),
            mode='lines'
        ))

        fig.update_layout(
            title="Sentiment Trend vs. Volume",
            xaxis_title="Date",
            yaxis=dict(title="Sentiment Score (-1 to +1)", range=[-1.1, 1.1]),
            yaxis2=dict(title="Volume (Count)", overlaying='y', side='right', showgrid=False),
            legend=dict(x=0, y=1.1, orientation='h'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Key Trend Insights")
        c1, c2 = st.columns(2)
        recent_avg = daily_trends['Smoothed_Sentiment'].iloc[-1]
        change = recent_avg - (daily_trends['Smoothed_Sentiment'].iloc[-30] if len(daily_trends) > 30 else daily_trends['Smoothed_Sentiment'].iloc[0])
        
        with c1:
            st.metric("Current Sentiment (7-Day)", f"{recent_avg:.2f}", delta=f"{change:.2f}")
        with c2:
            st.metric("Peak Volume Day", daily_trends.loc[daily_trends['Review_Volume'].idxmax()][date_column].strftime('%Y-%m-%d'))