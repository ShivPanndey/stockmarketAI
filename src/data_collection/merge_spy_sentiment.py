"""
merge_spy_sentiment.py

Merges daily sentiment features with SPY technical indicator data,
fixes NaN sentiment issues, and adds extra engineered features.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# === File Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

SPY_FILE = RAW_DIR / "SPY_Data.csv"  # Adjust if your SPY data path is different
DAILY_SENTIMENT_FILE = PROCESSED_DIR / "daily_sentiment.csv"
OUTPUT_FILE = PROCESSED_DIR / "spy_with_sentiment.csv"

# === Load SPY data ===
spy_df = pd.read_csv(SPY_FILE)
spy_df["date"] = pd.to_datetime(spy_df["timestamp"]).dt.tz_localize(None).dt.normalize()

# === Load sentiment data ===
sent_df = pd.read_csv(DAILY_SENTIMENT_FILE)
sent_df["date"] = pd.to_datetime(sent_df["date"]).dt.tz_localize(None).dt.normalize()

# Debug date ranges
print(f"SPY date range: {spy_df['date'].min()} → {spy_df['date'].max()}")
print(f"Sentiment date range: {sent_df['date'].min()} → {sent_df['date'].max()}")

# === Merge sentiment into SPY ===
merged_df = pd.merge(spy_df, sent_df, on="date", how="left")

# === Fill missing sentiment with 0 (no news effect) ===
sentiment_cols = ["positive", "neutral", "negative", "compound", "headline_count"]
merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(0)

# === Add Cyclic Encoding ===
merged_df["day_of_week"] = merged_df["date"].dt.dayofweek
merged_df["month"] = merged_df["date"].dt.month
merged_df["dow_sin"] = np.sin(2 * np.pi * merged_df["day_of_week"] / 7)
merged_df["dow_cos"] = np.cos(2 * np.pi * merged_df["day_of_week"] / 7)
merged_df["month_sin"] = np.sin(2 * np.pi * merged_df["month"] / 12)
merged_df["month_cos"] = np.cos(2 * np.pi * merged_df["month"] / 12)

# === Add Missing Lag / MA / Volatility Features ===
merged_df["Close_Lag_1"] = merged_df["close"].shift(1)
merged_df["Volume_Lag_1"] = merged_df["volume"].shift(1)
merged_df["RSI_Lag_1"] = merged_df["RSI_14"].shift(1)
merged_df["MA_5"] = merged_df["close"].rolling(5).mean()
merged_df["MA_10"] = merged_df["close"].rolling(10).mean()
merged_df["Volatility_5"] = merged_df["close"].rolling(5).std()

# === Momentum Features ===
for lag in [1, 3, 5, 10]:
    merged_df[f"momentum_{lag}"] = merged_df["close"] - merged_df["close"].shift(lag)
    merged_df[f"roc_{lag}"] = merged_df["close"].pct_change(lag)

# === Bollinger Band Width (20-day window) ===
rolling_mean = merged_df["close"].rolling(20).mean()
rolling_std = merged_df["close"].rolling(20).std()
upper_band = rolling_mean + (2 * rolling_std)
lower_band = rolling_mean - (2 * rolling_std)
merged_df["bb_width"] = upper_band - lower_band

# === Average True Range (ATR, 14-day) ===
high_low = merged_df["high"] - merged_df["low"]
high_close = np.abs(merged_df["high"] - merged_df["close"].shift())
low_close = np.abs(merged_df["low"] - merged_df["close"].shift())
tr = high_low.to_frame(name="tr")
tr["hc"] = high_close
tr["lc"] = low_close
tr["true_range"] = tr.max(axis=1)
merged_df["atr_14"] = tr["true_range"].rolling(14).mean()

# === Interaction Features ===
merged_df["sentiment_x_volume"] = merged_df["compound"] * merged_df["volume"]
merged_df["rsi_x_macd"] = merged_df["RSI_14"] * merged_df["MACD"]

# === Drop NaN from technical rolling calculations ===
merged_df.fillna(0, inplace=True)

# === Save merged dataset ===
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
merged_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Saved merged dataset with sentiment + engineered features → {OUTPUT_FILE}")
print(f"Shape: {merged_df.shape}")
