import pandas as pd
import json
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# --- CONFIGURATION ---
INPUT_FILE = "data/processed/youtube_sentiment.csv"
OUTPUT_FILE = "data/processed/topics_summary.json"
MODEL_NAME = "openai/gpt-oss-120b" 
BATCH_SIZE = 50  # Process 50 reviews at a time to fit in AI memory

def setup_chain():
    """Creates the LangChain pipeline."""
    llm = ChatGroq(
        temperature=0.2, 
        model_name=MODEL_NAME,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are a product analyst. Analyze the following customer reviews and extract sentiment ONLY for these specific topics:
        
        1. Battery
        2. Lifespan / Durability
        3. Overall Verdict (e.g., "watch is good", "worth buying")
        4. Performance
        5. Display
        6. Price

        Instructions:
        - DO NOT create new topics. Only use the 6 listed above.
        - If a review mentions "battery is bad", add the quote "battery is bad" to the 'negative' list for the Battery topic.
        - Ignore topics not in the list.

        Reviews to analyze:
        {text}

        Return the output as a valid JSON object with this EXACT structure:
        {{
            "topics": [
                {{
                    "name": "Battery",
                    "positive": [],
                    "negative": [],
                    "neutral": []
                }},
                {{
                    "name": "Performance",
                    "positive": [],
                    "negative": [],
                    "neutral": []
                }}
            ]
        }}
        """
    )
    return prompt | llm | JsonOutputParser()

def merge_results(main_data, new_data):
    """
    Combines the results from a new batch into the main list.
    """
    if not new_data or "topics" not in new_data:
        return main_data
    
    # Create a map for easy lookup: {"Battery": index, "Display": index}
    topic_map = {t["name"]: i for i, t in enumerate(main_data["topics"])}
    
    for new_topic in new_data["topics"]:
        name = new_topic.get("name")
        if name in topic_map:
            # Topic exists, extend the lists
            idx = topic_map[name]
            main_data["topics"][idx]["positive"].extend(new_topic.get("positive", []))
            main_data["topics"][idx]["negative"].extend(new_topic.get("negative", []))
            main_data["topics"][idx]["neutral"].extend(new_topic.get("neutral", []))
        else:
            # New topic (shouldn't happen with strict prompting, but safe to handle)
            main_data["topics"].append(new_topic)
            topic_map[name] = len(main_data["topics"]) - 1
            
    return main_data

def extract_topics_langchain():
    print("--- 🧐 Starting Full Topic Extraction (Batched) ---")

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    total_reviews = len(df)
    print(f"📊 Loaded {total_reviews} reviews to process.")

    # Initialize the structure for the final result
    final_results = {
        "topics": [
            {"name": "Battery", "positive": [], "negative": [], "neutral": []},
            {"name": "Lifespan / Durability", "positive": [], "negative": [], "neutral": []},
            {"name": "Overall Verdict", "positive": [], "negative": [], "neutral": []},
            {"name": "Performance", "positive": [], "negative": [], "neutral": []},
            {"name": "Display", "positive": [], "negative": [], "neutral": []},
            {"name": "Price", "positive": [], "negative": [], "neutral": []}
        ]
    }

    chain = setup_chain()
    
    # --- BATCH PROCESSING LOOP ---
    for i in range(0, total_reviews, BATCH_SIZE):
        batch = df['content'].astype(str).tolist()[i : i + BATCH_SIZE]
        batch_text = "\n".join(batch)
        
        print(f"   ⏳ Processing batch {i // BATCH_SIZE + 1} ({i} to {min(i + BATCH_SIZE, total_reviews)})...")
        
        try:
            # Analyze this chunk
            batch_result = chain.invoke({"text": batch_text})
            
            # Merge this chunk's answer into the main file
            final_results = merge_results(final_results, batch_result)
            
        except Exception as e:
            print(f"   ⚠️ Error on batch starting at {i}: {e}")
            continue

    # --- SAVE FINAL RESULTS ---
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ All batches complete. Saved to: {OUTPUT_FILE}")
    
    # Print Summary
    print("\n--- SUMMARY ---")
    for t in final_results["topics"]:
        name = t["name"]
        total = len(t["positive"]) + len(t["negative"]) + len(t["neutral"])
        if total > 0:
            print(f"📌 {name}: {len(t['positive'])} Pos, {len(t['negative'])} Neg")

if __name__ == "__main__":
    extract_topics_langchain()