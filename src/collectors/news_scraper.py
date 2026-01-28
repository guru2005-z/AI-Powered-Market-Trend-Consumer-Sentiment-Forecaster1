import json
import os
import time
import random
from GoogleNews import GoogleNews

def fetch_safe_news(query="boAt watch", limit=100):
    print(f"🐢 Safe-Fetching up to {limit} articles for '{query}'...")
    print("   (This will take time to avoid being blocked)")
    
    # 1. Initialize
    gn = GoogleNews(lang='en', region='IN', period='90d')
    
    # 2. Get Page 1
    try:
        gn.search(query)
        print(f"   ✅ Page 1 fetched")
    except Exception as e:
        print(f"   ❌ Critical Error on Page 1: {e}")
        return []

    all_results = gn.result()
    
    # 3. Loop for more pages
    # We loop gently. If we need 100 articles, we need ~10 pages.
    for page in range(2, 15):
        if len(all_results) >= limit:
            print("   🎯 Target limit reached!")
            break

        print(f"   ⏳ Sleeping... (Waiting for Google to relax)")
        # RANDOM SLEEP: Wait 5 to 10 seconds between requests. 
        # This is CRITICAL to avoid the 429 error.
        time.sleep(random.uniform(5, 10))
        
        try:
            gn.get_page(page)
            new_results = gn.result()
            
            # GoogleNews appends results to the internal list automatically,
            # but we check if we actually got new data.
            current_count = len(new_results)
            print(f"   ✅ Page {page} fetched. Total so far: {current_count}")
            
            all_results = new_results
            
        except Exception as e:
            print(f"   ⚠️ Blocked or Error on page {page}: {e}")
            print("   🛑 Stopping early to prevent ban extension.")
            break

    # 4. Deduplicate & Clean
    print(f"Processing {len(all_results)} raw results...")
    clean_articles = []
    seen_titles = set()

    for art in all_results:
        title = art.get('title')
        if title in seen_titles or len(clean_articles) >= limit:
            continue
        seen_titles.add(title)
        
        clean_articles.append({
            'title': title,
            'description': art.get('desc'), 
            'source': art.get('media'),    
            'url': art.get('link'),
            'published_at': art.get('date'),
            'content': art.get('desc')      
        })
            
    print(f"🏁 Final Count: {len(clean_articles)} unique articles.")
    return clean_articles

def save_to_json(data, filename="data/boat_news_safe.json"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"💾 Saved to {filename}")

if __name__ == "__main__":
    # LIMIT SET TO 100
    news = fetch_safe_news(query="boAt Smartwatch", limit=100)
    save_to_json(news)