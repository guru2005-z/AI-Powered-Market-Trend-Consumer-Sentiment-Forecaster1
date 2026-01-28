import os
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

def fetch_youtube_comments(query="boAt watch review", max_videos=100, max_comments=700):
    print(f"🔹 (YouTube) Searching for comments on: '{query}'...")
    print(f"   Target: {max_comments} comments from up to {max_videos} videos")
    
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        print(" Error: Missing YOUTUBE_API_KEY in .env")
        return pd.DataFrame()

    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        
        # 1. Search for videos
        search_response = youtube.search().list(
            q=query,
            part="id,snippet",
            maxResults=max_videos,
            type="video"
        ).execute()

        comments_list = []
        comments_collected = 0

        for item in search_response['items']:
            if comments_collected >= max_comments:
                break
                
            video_id = item['id']['videoId']
            video_title = item['snippet']['title']
            
            # 2. Get comments for each video
            try:
                # Calculate remaining comments needed
                remaining = max_comments - comments_collected
                batch_size = min(100, remaining)
                
                comment_response = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=batch_size, 
                    textFormat="plainText"
                ).execute()

                for comment in comment_response['items']:
                    if comments_collected >= max_comments:
                        break
                        
                    text = comment['snippet']['topLevelComment']['snippet']['textDisplay']
                    date = comment['snippet']['topLevelComment']['snippet']['publishedAt'][:10]
                    
                    comments_list.append({
                        "date": date,
                        "source": "YouTube",
                        "title": video_title,
                        "text": text,
                        "url": f"https://www.youtube.com/watch?v={video_id}"
                    })
                    comments_collected += 1
            except:
                continue 

        df = pd.DataFrame(comments_list)
        print(f" Success! Collected {len(df)} YouTube comments.")
        return df

    except Exception as e:
        print(f"YouTube Error: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Test Run - Scrape 700 comments
    df = fetch_youtube_comments("boAt watch review", max_comments=700)
    if not df.empty:
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv("data/raw/youtube_data.csv", index=False)
        print("📂 Saved to data/raw/youtube_data.csv")