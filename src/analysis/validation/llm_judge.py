import pandas as pd
import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
# REMOVED: from langchain.evaluation import load_evaluator (Not needed)
from langchain_core.prompts import PromptTemplate

load_dotenv()

# --- CONFIGURATION ---
STUDENT_FILE = "data/processed/youtube_sentiment.csv"
SAMPLE_SIZE = 50 # How many reviews to grade (don't grade all 1000, it's slow)

def run_auto_judge():
    print("--- ⚖️ Starting LLM-as-a-Judge Validation ---")
    
    if not os.path.exists(STUDENT_FILE):
        print(f"❌ Error: {STUDENT_FILE} not found.")
        return

    # 1. Load the Student's Work
    df = pd.read_csv(STUDENT_FILE)
    
    # We only validate a random sample to save time
    if len(df) > SAMPLE_SIZE:
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=42).copy()
    else:
        sample_df = df.copy()
        
    print(f"👨‍⚖️ The Judge (Llama3-70b) will evaluate {len(sample_df)} reviews...\n")

    # 2. Initialize the Judge (Must be a stronger model)
    # We use 70b because it is smarter than the 8b model used for the main task
    judge_llm = ChatGroq(
        temperature=0,
        model_name="openai/gpt-oss-120b", 
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # 3. Validation Loop
    correct_count = 0
    results = []

    print(f"{'REVIEW':<50} | {'STUDENT':<10} | {'JUDGE':<10}")
    print("-" * 80)

    for index, row in sample_df.iterrows():
        # Get data safely
        review_text = str(row.get('content', ''))[:300] 
        student_answer = str(row.get('sentiment', '0')).lower().strip()
        
        # We ask the Judge to verify
        eval_prompt = f"""
        You are a strict teacher grading a student's homework.
        
        Review: "{review_text}"
        Student's Answer: "{student_answer}" (Note: 1=Positive, -1=Negative, 0=Neutral)
        
        Task:
        1. Analyze the review yourself.
        2. Decide if the Student's Answer is Correct (yes/no).
        3. If wrong, provide the correct sentiment (1, -1, or 0).
        
        Return ONLY a JSON object:
        {{
            "correct": true,
            "judge_sentiment": "1",
            "reason": "short explanation"
        }}
        """
        
        try:
            # Direct invocation 
            response = judge_llm.invoke(eval_prompt).content
            
            # Simple parsing 
            clean_json = response.replace("```json", "").replace("```", "").strip()
            grade = json.loads(clean_json)
            
            is_correct = grade.get("correct", False)
            judge_sentiment = grade.get("judge_sentiment", "unknown")
            
            if is_correct:
                correct_count += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
                
            print(f"{review_text[:47]:<47}... | {student_answer:<10} | {status}")
            
            results.append({
                "review": review_text,
                "student": student_answer,
                "judge": judge_sentiment,
                "correct": is_correct,
                "reason": grade.get("reason")
            })

        except Exception as e:
            print(f"⚠️ Error judging row {index}: {e}")

    # 4. Final Report
    if len(sample_df) > 0:
        score = (correct_count / len(sample_df)) * 100
        print("\n" + "="*40)
        print(f"🏆 FINAL AUTO-VALIDATION SCORE: {score:.1f}%")
        print("="*40)
        
        # Save the Judge's Report
        report_file = "data/processed/judge_report.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Detailed Judge Report saved to: {report_file}")

if __name__ == "__main__":
    run_auto_judge()