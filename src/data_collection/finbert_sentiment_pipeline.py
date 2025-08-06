"""
finbert_sentiment_pipeline.py

Best-practice sentiment analysis pipeline for financial news using FinBERT.
Requires:
    - Python 3.10+
    - PyTorch >= 2.6 (Linux or compatible macOS)
    - transformers
    - pandas
"""

import pandas as pd
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "ProsusAI/finbert"

def load_model():
    """Load FinBERT model and tokenizer."""
    print("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

def get_sentiment(text, tokenizer, model):
    """Get sentiment scores for a single text."""
    if not isinstance(text, str) or not text.strip():
        return {"label": "neutral", "positive": 0, "neutral": 1, "negative": 0, "compound": 0}

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = softmax(outputs.logits, dim=-1).numpy()[0]
    pred_class = probs.argmax()
    label_map = model.config.id2label

    return {
        "label": label_map[pred_class].lower(),
        "positive": probs[2],
        "neutral": probs[1],
        "negative": probs[0],
        "compound": probs[2] - probs[0]
    }

def process_headlines(input_csv, output_csv):
    """Process CSV of headlines and save sentiment results."""
    tokenizer, model = load_model()
    df = pd.read_csv(input_csv)

    sentiments = df["headline"].apply(lambda t: get_sentiment(t, tokenizer, model))
    df_sent = pd.json_normalize(sentiments)
    df_combined = pd.concat([df, df_sent], axis=1)

    df_combined.to_csv(output_csv, index=False)
    print(f"✅ Sentiment analysis complete. Saved to {output_csv}")

if __name__ == "__main__":
    process_headlines("news_headlines.csv", "news_with_sentiment.csv")
