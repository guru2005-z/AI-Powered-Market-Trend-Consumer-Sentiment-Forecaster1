import pandas as pd
import os
import time
import concurrent.futures
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- CONFIGURATION ---
MODEL_NAME = "openai/gpt-oss-120b" 
MAX_WORKERS = 5  # Process 5 reviews at the same time

def setup_chain():
    """Creates the LangChain processing pipeline."""
    llm = ChatGroq(
        temperature=0.0,
        model_name=MODEL_NAME,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    prompt = ChatPromptTemplate.from_template(
        """
        Classify the sentiment of this text as either '1' (Positive), '-1' (Negative), or '0' (Neutral).
        return ONLY the number.
        
        Text: "{text}"
        """
    )
    
    return prompt | llm | StrOutputParser()

def analyze_single_review(text, chain):
    """Worker function to analyze one review."""
    try:
        # Clean text briefly
        clean_text = str(text)[:300] 
        response = chain.invoke({"text": clean_text})
        
        # Clean response to get just the number
        sentiment = response.strip()
        if "1" in sentiment and "-1" not in sentiment: return 1
        if "-1" in sentiment: return -1
        if "0" in sentiment: return 0
        return 0 # Default to neutral if unsure
        
    except Exception:
        return 0

def run_fast_sentiment(input_file, output_file):
    print(f"--- 🚀 Starting Fast Sentiment Analysis: {input_file} ---")
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return

    df = pd.read_csv(input_file)
    print(f"📂 Loaded {len(df)} rows.")

    # Detect text column
    text_col = next((col for col in ['content', 'text', 'title'] if col in df.columns), None)
    if not text_col:
        print("❌ No text column found.")
        return

    # Setup LangChain
    chain = setup_chain()
    
    print(f"⚡ Processing with {MAX_WORKERS} parallel workers...")
    start_time = time.time()
    
    results = []
    
    # --- PARALLEL EXECUTION ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a list of tasks
        future_to_index = {
            executor.submit(analyze_single_review, row[text_col], chain): i 
            for i, row in df.iterrows()
        }
        
        # Collect results as they finish
        completed = 0
        results_map = {}
        
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            res = future.result()
            results_map[index] = res
            
            completed += 1
            if completed % 50 == 0:
                print(f"   ...Processed {completed}/{len(df)} reviews")

    # Reassemble results in correct order
    final_sentiments = [results_map[i] for i in range(len(df))]
    
    # Save
    df['sentiment'] = final_sentiments
    df.to_csv(output_file, index=False)
    
    duration = time.time() - start_time
    print(f"✅ Done in {duration:.2f} seconds.")
    print(f"💾 Saved to {output_file}\n")

if __name__ == "__main__":
    # Define your files here
    files = [
        ("data/processed/youtube_cleaned.csv", "data/processed/youtube_sentiment.csv"),
        # Add boat news file if you have it
        ("data/processed/boat_news_raw_cleaned.csv", "data/processed/boat_news_raw_sentiment.csv") 
    ]
    
    for in_f, out_f in files:
        run_fast_sentiment(in_f, out_f)