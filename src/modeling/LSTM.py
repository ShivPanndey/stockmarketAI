"""
LSTM.py - LSTM model training using SPY technical indicators + sentiment + cyclic date features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# === File Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "spy_with_sentiment.csv"

# Load dataset
df = pd.read_csv(PROCESSED_DATA)

# Ensure datetime format
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
else:
    df["date"] = pd.to_datetime(df["timestamp"])

# Create target
df["Target"] = (df["close"].shift(-1) > df["close"]).astype(int)

# Fill missing sentiment values BEFORE dropping
sentiment_cols = ["positive", "neutral", "negative", "compound", "headline_count"]
df[sentiment_cols] = df[sentiment_cols].fillna(0)

# Cyclic encoding for day of week and month
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Drop only rows missing essential values
df.dropna(subset=["close", "Target"], inplace=True)

# Features list
features = [
    "close", "volume", "EMA_50", "EMA_200",
    "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
    "Close_Lag_1", "Volume_Lag_1", "RSI_Lag_1",
    "MA_5", "MA_10", "Volatility_5",
    "positive", "neutral", "negative", "compound", "headline_count",
    "dow_sin", "dow_cos", "month_sin", "month_cos"
]

X = df[features]
y = df["Target"]

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Sequence builder
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y.values)

# Train/test split
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# Build LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
model.fit(X_train, y_train, epochs=30, batch_size=32,
          validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)

# Evaluate
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("LSTM Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
