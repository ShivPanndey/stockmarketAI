from alpaca_trade_api.rest import REST, TimeFrame
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

# Connect to Alpaca API
api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Fetch historical data for SPY ETF
spy_bars = api.get_bars(
    "SPY",
    timeframe=TimeFrame.Day,
    start="2022-01-01",
    end="2024-12-31",
).df

spy_bars = spy_bars.sort_index()

# Add technical indicators
spy_bars["EMA_50"] = ta.ema(spy_bars["close"], length=50)
spy_bars["EMA_200"] = ta.ema(spy_bars["close"], length=200)
spy_bars["RSI_14"] = ta.rsi(spy_bars["close"], length=14)
macd = ta.macd(spy_bars["close"], fast=12, slow=26, signal=9)
spy_bars["MACD"] = macd["MACD_12_26_9"]
spy_bars["MACD_Signal"] = macd["MACDs_12_26_9"]
spy_bars["MACD_Hist"] = macd["MACDh_12_26_9"]

# === Predictive Features ===
# Lag features
spy_bars["Close_Lag_1"] = spy_bars["close"].shift(1)
spy_bars["Volume_Lag_1"] = spy_bars["volume"].shift(1)
spy_bars["RSI_Lag_1"] = spy_bars["RSI_14"].shift(1)

# Rolling means
spy_bars["MA_5"] = spy_bars["close"].rolling(window=5).mean()
spy_bars["MA_10"] = spy_bars["close"].rolling(window=10).mean()

# Volatility (rolling std dev of returns)
spy_bars["Returns"] = spy_bars["close"].pct_change()
spy_bars["Volatility_5"] = spy_bars["Returns"].rolling(window=5).std()

# Drop rows with NaNs from indicators/lags
spy_bars.dropna(inplace=True)

# Save enhanced data
spy_bars.to_csv("SPY_Data_Enhanced.csv")
