# 🔗 merge_spy_sentiment.py

## 📌 Purpose
Merges:
- **SPY technical data** from `SPY_Data.csv`
- **Daily aggregated sentiment** from `daily_sentiment.csv`

Also:
- Fills missing sentiment with neutral values.
- Adds engineered technical features for modeling.

---

## ⚙️ Workflow
1. **Load datasets**
   - Reads raw SPY OHLC data (`data/raw/SPY_Data.csv`).
   - Reads daily sentiment data (`data/processed/daily_sentiment.csv`).
   - Normalizes dates for consistent merging.
2. **Merge datasets**
   - Left-join sentiment to SPY data on `date`.
   - Missing sentiment → `0`.
3. **Add cyclic encoding**
   - `dow_sin`, `dow_cos`, `month_sin`, `month_cos`.
4. **Add lag features**
   - Previous-day close, volume, RSI (`Close_Lag_1`, `Volume_Lag_1`, `RSI_Lag_1`).
5. **Add moving averages**
   - 5-day and 10-day (`MA_5`, `MA_10`).
6. **Add volatility**
   - 5-day rolling standard deviation (`Volatility_5`).
7. **Momentum features**
   - Price momentum and ROC for various lags.
8. **Bollinger Band width**
   - 20-day rolling calculation.
9. **Average True Range**
   - 14-day volatility measure.
10. **Interaction features**
    - Sentiment × Volume, RSI × MACD.
11. **Save processed dataset**
    - Outputs `spy_with_sentiment.csv` for modeling.

---

## 🛠 Key Outputs
- `data/processed/spy_with_sentiment.csv`
- Shape and column info printed after processing.

---

## 🔮 Future Improvements
- Add more sentiment interaction terms.
- Test different rolling window lengths for volatility measures.
- Integrate macroeconomic indicators.
