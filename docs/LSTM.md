# 🔮 LSTM.py

## 📌 Purpose
Trains a **Long Short-Term Memory (LSTM)** deep learning model to predict **next-day SPY price direction** using:
- Sequential **technical indicators**
- **Sentiment analysis** features
- Cyclic seasonal patterns

---

## ⚙️ Workflow
1. **Load processed dataset**
   - Reads `data/processed/spy_with_sentiment.csv`.
   - Parses date column.
2. **Create target variable**
   - Binary target: `1` if tomorrow's close > today's close, else `0`.
3. **Fill missing sentiment values**
   - Default neutral sentiment values for missing days.
4. **Add cyclic encoding**
   - Converts day-of-week and month into sine/cosine components.
5. **Scaling**
   - `MinMaxScaler` applied before sequence creation.
6. **Sequence creation**
   - Converts time-series data into overlapping sequences for LSTM input.
   - Example: 10-day lookback window to predict the next day.
7. **Model architecture**
   - Input layer → LSTM layer(s) → Dropout → Dense output layer.
   - Sigmoid activation for binary classification.
8. **Training**
   - Binary crossentropy loss.
   - Adam optimizer.
   - Early stopping to prevent overfitting.
9. **Evaluation**
   - Accuracy score.
   - Classification report.
   - Confusion matrix.

---

## 🛠 Key Parameters
- **lookback window**: 10 days
- **LSTM units**: 50–100
- **dropout rate**: 0.2–0.5
- **batch size**: 32–64
- **epochs**: 50–200 (early stopping enabled)

---

## 📈 Output
- Classification report.
- Confusion matrix.
- Accuracy score.

---

## 🔮 Future Improvements
- Add attention mechanism for time-series weighting.
- Tune sequence length for optimal results.
- Experiment with bidirectional LSTMs.
