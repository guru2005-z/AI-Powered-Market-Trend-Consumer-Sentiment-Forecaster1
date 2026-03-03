# 📈 AI-Powered Market Trend & Consumer Sentiment Forecaster

An enterprise-grade, end-to-end AI application that ingests unstructured consumer feedback, processes it through a Retrieval-Augmented Generation (RAG) pipeline, and provides real-time market insights via an interactive dashboard. 

This tool is designed for Product Managers and Market Analysts to track brand health, understand consumer pain points, and generate automated executive intelligence reports.

---

## 🚀 Key Features

### 1. 🤖 Context-Aware RAG Chatbot
* **Intelligent Q&A:** Ask complex questions about market data (e.g., *"What are the main complaints about battery life?"*).
* **Powered by Gemini 2.5 Flash:** Utilizes Google's generative AI, grounded securely in your specific data using LangChain.

### 2. 📊 Interactive Analytics Dashboard
* **Sentiment Visualization:** Dynamic Plotly pie charts and bar graphs displaying positive, negative, and neutral sentiment distributions.
* **Time-Series Trend Tracking:** Analyze how consumer sentiment and review volume fluctuate over time to identify product launch impacts or PR crises.
* **Feature-Level Scoring:** Breaks down sentiment by specific product features (e.g., Display, Performance, Price).

### 3. 🧠 Robust Vector Database (Pinecone)
* **Local Embeddings:** Uses HuggingFace's `all-MiniLM-L6-v2` (`sentence-transformers`) for fast, cost-free local semantic embedding.
* **Cloud Storage:** Upserts 384-dimensional vectors to a Pinecone Serverless Index for high-speed similarity search (`cosine` metric).

### 4. 🧹 Automated Data ETL Pipeline
* **Aggressive Data Cleaning:** Custom Python scripts that strip HTML, URLs, emojis, and special characters from raw scraped data.
* **Standardization:** Automatically standardizes dates, cleans currency/pricing strings, and normalizes review ratings.

### 5. 🔔 Smart Alerts & Automated Executive Reporting
* **Health Monitoring:** Scans recent data for sentiment drops (e.g., average sentiment falling below -0.20) and triggers UI alerts.
* **AI Report Generation:** Automatically drafts professional, HTML-formatted Executive Market Reports based on the current data context.
* **Email Integration:** Securely emails the generated reports directly to stakeholders via `yagmail` SMTP integration.

---

## 🏗️ System Architecture & Workflow

1. **Data Collection (`src/collectors/`):** Raw data is scraped from YouTube comments, news articles, or product reviews and saved as JSON/CSV.
2. **Data Processing (`src/processing/`):** The `cleaner.py` script purges noise, normalizes metrics, and exports clean datasets to `data/processed/`.
3. **Vectorization (`src/ui/build_db.py`):** Cleaned text is chunked into LangChain `Document` objects, converted into semantic vectors via HuggingFace, and uploaded to Pinecone.
4. **Retrieval & UI (`src/ui/app.py`):** The Streamlit frontend loads the interactive charts. When a user asks a question, the app queries Pinecone for the most relevant context and passes it to the Gemini LLM for a natural language response.

---

## 🛠️ Technology Stack

| Category | Technologies |
| :--- | :--- |
| **Frontend/UI** | Streamlit, Plotly (Express & Graph Objects) |
| **AI & LLM** | Google Gemini 2.5 Flash, LangChain (v0.1.20) |
| **Embeddings** | HuggingFace (`sentence-transformers==3.0.1`) |
| **Vector Database** | Pinecone Client (Serverless) |
| **Data Engineering** | Pandas, Regex (re), JSON |
| **Utilities** | Python-dotenv, Yagmail (SMTP) |

---

## 📂 Project Structure

```text
AI_Market_Forecaster/
├── .env                        # Environment variables (API Keys)
├── requirements.txt            # Strictly pinned dependency versions
├── data/
│   ├── raw/                    # Unprocessed data (e.g., boat_news_raw.json)
│   └── processed/              # Cleaned, ML-ready CSVs and JSONs
└── src/
    ├── analysis/               # Sentiment logic & topic modeling
    │   ├── validation/
    │   │   └── llm_judge.py
    │   ├── sentiment.py
    │   └── topics.py
    ├── collectors/             # Web scrapers (YouTube, News)
    │   ├── news_scraper.py
    │   └── youtube_scraper.py
    ├── processing/             # Data cleaning, deduplication, and formatting
    │   ├── chunker.py
    │   └── cleaner.py
    ├── rag_engine/             # Core LangChain Retrieval mechanics
    │   └── build_db.py         # Pinecone Index creation & Vector upload script
    └── ui/
        └── app.py              # Main Streamlit Application & UI logic
        
⚙️ Installation & Local Setup
Prerequisites
Python 3.10 or higher

Git

A Google Gemini API Key

A Pinecone API Key (Free tier works)

A Gmail account with an App Password (for email reporting)

1. Clone the Repository
Bash
git clone [https://github.com/guru2005-z/AI-Powered-Market-Trend-Consumer-Sentiment-Forecaster1.git](https://github.com/guru2005-z/AI-Powered-Market-Trend-Consumer-Sentiment-Forecaster1.git)
cd AI-Powered-Market-Trend-Consumer-Sentiment-Forecaster1
2. Create a Virtual Environment (Recommended)
Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install Dependencies
Note: This project relies on strictly pinned versions of LangChain and Sentence Transformers to ensure system stability.

Bash
pip install -r requirements.txt
4. Configure Environment Variables
Create a file named .env in the root directory and add your credentials:

Code snippet
# AI & Vector DB Keys
GOOGLE_API_KEY="your_google_gemini_api_key"
PINECONE_API_KEY="your_pinecone_api_key"

# Email Configuration (For Sending Executive Reports)
# Note: You must generate a 16-digit "App Password" in your Google Account Settings.
GMAIL_USER="your_email@gmail.com"
GMAIL_APP_PASSWORD="your_16_digit_app_password"
🚀 Usage Guide
Step 1: Process Raw Data
If you have new raw data in data/raw/, run the data processing pipeline to clean and standardize it:

Bash
python src/processing/cleaner.py
Step 2: Build the AI Knowledge Base (Pinecone)
Run the database builder to vectorize your cleaned data and upload it to the cloud. You only need to run this once (or whenever you add new data).

Bash
python src/ui/build_db.py
Wait for the console to output ✅ SUCCESS! Database Built before proceeding.

Step 3: Launch the Dashboard
Start the interactive Streamlit server:

Bash
streamlit run src/ui/app.py
Navigate to http://localhost:8501 in your web browser to view the application.

📸 Application Previews
(Add screenshots of your application here by uploading them to an assets/ folder in your repo and linking them below)

Market Overview: ![Overview Dashboard](link_to_image)

Trend Analytics: ![Trend Graphs](link_to_image)

RAG Chatbot: ![Chatbot Interface](link_to_image)

🔮 Future Enhancements
Live Social Listening: Implement real-time streaming APIs for Twitter/X and Reddit.

Multi-Agent Architecture: Add specialized AI agents (e.g., a "Pricing Analyst" vs. a "PR Analyst").

Advanced Authentication: Add user login screens to secure executive reports.