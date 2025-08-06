"""
news_sentiment_mac.py

Fetches historical SPY-related financial news headlines (up to 6+ months back)
and performs sentiment analysis using a lightweight financial sentiment model.

This macOS-compatible version avoids large model dependencies like full FinBERT.
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch
import time
import os
from dotenv import load_dotenv

# === Load API key from .env ===
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")  # Make sure your .env has: NEWS_API_KEY=your_key_here

if not API_KEY:
    raise ValueError("❌ NEWS_API_KEY not found in .env file. Please add it and try again.")

# === CONFIG ===
QUERY = "SPY OR S&P 500 OR stock market"
MONTHS_BACK = 6  # How many months of news to fetch
BATCH_DELAY = 1  # seconds delay between API calls (avoid rate limits)

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "news_with_sentiment.csv"

# === Load lightweight sentiment model ===
tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

label_map = {0: "negative", 1: "neutral", 2: "positive"}

def get_sentiment(text):
    """Run sentiment analysis on a headline."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=-1).detach().numpy()[0]

    return {
        "positive": float(probs[2]),
        "neutral": float(probs[1]),
        "negative": float(probs[0]),
        "compound": float(probs[2] - probs[0])
    }

def fetch_news(from_date, to_date):
    """Fetch news headlines for a given date range."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": QUERY,
        "from": from_date.strftime("%Y-%m-%d"),
        "to": to_date.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "relevancy",
        "apiKey": API_KEY,
        "pageSize": 100
    }
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "ok":
        print(f"❌ Failed for {from_date} → {to_date}: {data}")
        return []

    return [{"date": article["publishedAt"][:10], "headline": article["title"]}
            for article in data.get("articles", [])]

# === Main loop: Fetch month-by-month ===
all_data = []
today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
current_end = today

for _ in range(MONTHS_BACK):
    current_start = (current_end - timedelta(days=30))
    print(f"📅 Fetching {current_start.date()} → {current_end.date()}")

    headlines = fetch_news(current_start, current_end)
    print(f"   Found {len(headlines)} articles")

    # Add to all_data
    all_data.extend(headlines)

    # Move window back one month
    current_end = current_start - timedelta(days=1)
    time.sleep(BATCH_DELAY)

# === Convert to DataFrame ===
df = pd.DataFrame(all_data)

if df.empty:
    print("❌ No news data collected. Check your API key or query.")
else:
    # === Run sentiment analysis ===
    sentiments = df["headline"].apply(get_sentiment)
    sentiment_df = pd.DataFrame(list(sentiments))
    df = pd.concat([df, sentiment_df], axis=1)

    # Save output
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved news_with_sentiment.csv → {OUTPUT_FILE}")
    print(f"Rows: {len(df)}, Date range: {df['date'].min()} → {df['date'].max()}")
