"""
XGBoost_SPY.py - XGBoost model training using SPY technical indicators + sentiment + cyclic date features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

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

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

# Handle imbalance
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Model + hyperparameter tuning
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", scale_pos_weight=pos_weight)
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}
search = RandomizedSearchCV(
    xgb, param_distributions=param_grid,
    n_iter=10, scoring="accuracy", cv=3, verbose=1, n_jobs=-1
)
search.fit(X_train, y_train)

# Best model & evaluation
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

print("XGBoost Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", best_model.score(X_test, y_test))
