"""
aggregate_sentiment.py

Aggregates per-headline sentiment into daily averages for merging with SPY data.
"""

import pandas as pd
from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

NEWS_FILE = PROCESSED_DIR / "news_with_sentiment.csv"
OUTPUT_FILE = PROCESSED_DIR / "daily_sentiment.csv"

# Load sentiment CSV
news_df = pd.read_csv(NEWS_FILE)

# Ensure 'date' is in datetime format
news_df["date"] = pd.to_datetime(news_df["date"])

# Aggregate by date: mean sentiment scores
daily_sentiment = (
    news_df.groupby("date")[["positive", "neutral", "negative", "compound"]]
    .mean()
    .reset_index()
)

# Also count number of headlines per day
daily_sentiment["headline_count"] = (
    news_df.groupby("date")["headline"].count().values
)

# Save output
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
daily_sentiment.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Saved daily_sentiment.csv with aggregated sentiment features → {OUTPUT_FILE}")
