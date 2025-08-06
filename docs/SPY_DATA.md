# 📊 SPY_Data.py

## 📌 Purpose
Collects **SPY historical OHLCV data** and generates **technical indicators** for use in predictive models.

---

## ⚙️ Workflow
1. **Fetch historical SPY data**
   - Retrieves price and volume data using:
     - Alpaca API
     - Or from stored CSV in `data/raw/SPY_Data.csv`
2. **Ensure datetime consistency**
   - Parses timestamps and sets correct time zones if needed.
3. **Generate technical indicators**
   - **Moving Averages**
     - EMA_50, EMA_200
   - **Relative Strength Index (RSI)**
     - RSI_14
   - **MACD**
     - MACD line, MACD signal, MACD histogram
   - **Rolling Volatility**
     - Standard deviation over rolling windows
4. **(Optional) Lag Features**
   - Previous-day close, RSI, volume.
5. **Save raw dataset**
   - Outputs to `data/raw/SPY_Data.csv` for later processing.

---

## 🛠 Key Outputs
- **`SPY_Data.csv`**
  - OHLCV data + technical indicators.
- Input to `merge_spy_sentiment.py`.

---

## 📈 Example Output
| timestamp           | open   | high   | low    | close  | volume | EMA_50 | EMA_200 | RSI_14 | MACD | MACD_Signal | MACD_Hist |
|---------------------|--------|--------|--------|--------|--------|--------|---------|--------|------|-------------|-----------|
| 2025-07-10 09:30:00 | 448.23 | 449.10 | 447.50 | 448.75 | 3.2M   | 447.80 | 442.50  | 54.23  | 0.15 | 0.12        | 0.03      |

---

## 🔮 Future Improvements
- Add Bollinger Bands.
- Integrate VIX volatility index as an additional feature.
- Automate regular SPY data refresh for live predictions.
