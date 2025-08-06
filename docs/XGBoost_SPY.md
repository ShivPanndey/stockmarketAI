# 🐍 XGBoost_SPY.py

## 📌 Purpose
Trains an **XGBoost classification model** to predict **next-day SPY price direction** based on:
- **Technical indicators** (EMA, RSI, MACD, moving averages, volatility)
- **Sentiment features** (daily aggregated scores from news headlines)
- **Cyclic date encodings** (day-of-week and month seasonality)

---

## ⚙️ Workflow
1. **Load processed dataset**
   - Reads `data/processed/spy_with_sentiment.csv`.
   - Ensures dates are properly parsed.
2. **Create target variable**
   - Binary target: `1` if tomorrow's close > today's close, else `0`.
3. **Fill missing sentiment**
   - Replace NaN sentiment values with neutral defaults.
4. **Add cyclic encoding**
   - `dow_sin`, `dow_cos` for days of the week.
   - `month_sin`, `month_cos` for months of the year.
5. **Feature selection**
   - Uses both technical and sentiment features.
6. **Scaling**
   - `MinMaxScaler` applied to all features.
7. **Train-test split**
   - 80% train, 20% test (chronological order preserved).
8. **Class imbalance handling**
   - Adjusts `scale_pos_weight` in XGBoost based on class ratio.
9. **Hyperparameter tuning**
   - `RandomizedSearchCV` for parameters like learning rate, depth, and tree subsampling.
10. **Model training & evaluation**
    - Reports accuracy, precision, recall, F1-score.
    - Displays confusion matrix.

---

## 🛠 Key Parameters
- **n_estimators**: 100–300
- **max_depth**: 3–7
- **learning_rate**: 0.01–0.1
- **subsample**: 0.8–1.0
- **colsample_bytree**: 0.8–1.0

---

## 📈 Output
- Classification report.
- Confusion matrix.
- Accuracy score.

---

## 🔮 Future Improvements
- Add more lagged sentiment features.
- Try gradient boosting variants (LightGBM, CatBoost).
- Integrate feature importance visualization.
