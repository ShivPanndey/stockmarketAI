# 📈 Stock Market AI — Hybrid Technical & Sentiment Model for SPY

## 📌 Overview
This project predicts **SPY stock price movements** using a **hybrid approach**:
- **Technical Indicators** — Derived from SPY historical price and volume data.
- **Sentiment Analysis** — Daily aggregated sentiment from financial news headlines.

The combination of **quantitative signals** (technical) and **qualitative signals** (sentiment) aims to improve predictive accuracy compared to using technicals alone.

---

## 🚀 Features
- Automated **SPY historical data collection** via Alpaca API.
- **Financial news ingestion** and sentiment scoring via financial-domain transformer models.
- Aggregated **daily sentiment scores** merged with SPY technical indicators.
- **Two modeling approaches**:
  - **LSTM** — Sequential deep learning model for time-series forecasting.
  - **XGBoost** — Gradient boosting model for tabular classification.
- Modular, reproducible pipeline.

---

## 🗂 Project Structure

```bash
stockMarketAI/
├── data/
│   ├── raw/                          # Original market & news headline data
│   │   ├── news_headlines.csv
│   │   └── SPY_Data.csv
│   ├── processed/                    # Cleaned, aggregated, feature-engineered datasets
│   │   ├── news_with_sentiment.csv
│   │   ├── daily_sentiment.csv
│   │   └── spy_with_sentiment.csv
│
├── docs/                             # Documentation for each module
│   ├── AGGREGATE_SENTIMENT.md        # Aggregating daily sentiment
│   ├── FINBERT_SENTIMENT.md          # Sentiment pipeline explanation
│   ├── LSTM.md                       # LSTM model documentation
│   ├── MERGE_SPY_SENTIMENT.md        # Merging SPY + sentiment data
│   ├── SPY_DATA.md                   # SPY data & technical indicators
│   └── XGBoost_SPY.md                # XGBoost model documentation
│
├── notebooks/                        # Jupyter notebooks for exploration & testing
│   ├── LSTM.ipynb
│   └── XGBoost_SPY.ipynb
│
├── src/
│   ├── data_collection/              # Scripts for fetching & preparing data
│   │   ├── aggregate_sentiment.py    # Aggregate sentiment by day
│   │   ├── finbert_sentiment_pipeline.py  # High-performance FinBERT sentiment analysis
│   │   ├── merge_spy_sentiment.py    # Merge technical & sentiment features
│   │   ├── News_data.py              # News API fetching
│   │   ├── news_sentiment_mac.py     # MacOS-compatible sentiment pipeline
│   │   └── SPY_Data.py               # SPY historical + technical indicators
│   │
│   └── modeling/                     # Machine learning model scripts
│       ├── LSTM.py                   # LSTM deep learning model
│       └── XGBoost_SPY.py            # XGBoost gradient boosting model
│
├── README.md                         # Main project overview
├── requirements.txt                  # Python dependencies
├── LICENSE                           # Project license
└── __init__.py                       # Python package initializer
```
---

## 📊 Workflow

### **1. Collect SPY Historical Data**
- `src/data_collection/SPY_Data.py` → Generates technical indicators for SPY.

### **2. Collect Financial News Data**
- `src/data_collection/news_sentiment_mac.py` → Fetches financial news headlines.

### **3. Sentiment Analysis**
- `src/data_collection/finbert_sentiment_pipeline.py` (Linux / modern macOS).
- `src/data_collection/news_sentiment_mac.py` (macOS fallback).
- Outputs: `news_with_sentiment.csv`.

### **4. Aggregate Sentiment by Day**
- `src/data_collection/aggregate_sentiment.py` → `daily_sentiment.csv`.

### **5. Merge Sentiment + Technical Data**
- `src/data_collection/merge_spy_sentiment.py` → `spy_with_sentiment.csv`.

### **6. Train Models**
- `src/modeling/LSTM.py` — Sequential deep learning.
- `src/modeling/XGBoost_SPY.py` — Gradient boosting model.

---

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/stockMarketAI.git
cd stockMarketAI

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
---

## 📚 Documentation

| File / Script                                               | Description                                                                                              |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [**SPY\_Data.py**](docs/SPY_DATA.md)                        | Collects SPY historical OHLCV data and generates technical indicators (EMA, RSI, MACD, volatility).      |
| [**aggregate\_sentiment.py**](docs/AGGREGATE_SENTIMENT.md)  | Aggregates per-headline sentiment scores into daily averages for merging with SPY data.                  |
| [**merge\_spy\_sentiment.py**](docs/MERGE_SPY_SENTIMENT.md) | Merges SPY technical indicators with aggregated sentiment data and adds engineered features.             |
| [**XGBoost\_SPY.py**](docs/XGBoost_SPY.md)                  | Trains an XGBoost classification model using technical + sentiment features for SPY movement prediction. |
| [**LSTM.py**](docs/LSTM.md)                                 | Trains a sequential LSTM deep learning model for time-series forecasting of SPY price direction.         |

## 📈 Model Performance
```bash
XGBoost Classification Report:
               precision    recall  f1-score   support

           0       0.48      0.71      0.58        45
           1       0.71      0.48      0.58        66

    accuracy                           0.58       111
   macro avg       0.60      0.60      0.58       111
weighted avg       0.62      0.58      0.58       111

Confusion Matrix:
 [[32 13]
 [34 32]]

LSTM Classification Report:
               precision    recall  f1-score   support

           0       0.00      0.00      0.00        45
           1       0.59      1.00      0.74        64

    accuracy                           0.59       109
   macro avg       0.29      0.50      0.37       109
weighted avg       0.34      0.59      0.43       109

Confusion Matrix:
 [[ 0 45]
 [ 0 64]]
```

## Future Improvements
Extend historical sentiment coverage via alternative news sources (Google News, Yahoo Finance).
Add more advanced NLP models such as FinBERT-large or finance-tuned GPT models for sentiment scoring.
Explore hybrid ensemble models combining LSTM, XGBoost, and attention-based architectures.
Deploy as a live dashboard with daily retraining using Streamlit.

## 📜 License
This project is licensed under the MIT License — see the LICENSE file for details.



