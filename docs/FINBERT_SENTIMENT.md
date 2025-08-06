# FinBERT-Based Financial News Sentiment Analysis

## Overview
This script (`finbert_sentiment_pipeline.py`) implements a **best-practice sentiment analysis pipeline** for financial news using **FinBERT** (`ProsusAI/finbert`), a state-of-the-art transformer model fine-tuned on financial text.

It processes a CSV file of daily financial news headlines (e.g., from NewsAPI, Google News, or other feeds) and outputs:
- **label**: predicted sentiment class (`positive`, `neutral`, `negative`)
- **positive**, **neutral**, **negative**: probability scores
- **compound**: sentiment polarity score = positive − negative

## Why This Approach is Considered Best
1. **Domain-Specific Model**  
   FinBERT is trained specifically on **financial language** (earnings reports, analyst commentary, market news), making it far more accurate than generic sentiment models.
   
2. **Quantitative Output for Modeling**  
   Instead of only a label, the pipeline provides **continuous sentiment scores** (`positive`, `neutral`, `negative`) and a **compound score** that can be used directly as numerical features in predictive models.

3. **Daily Aggregation Ready**  
   The generated sentiment scores can be aggregated **by day** and merged with technical indicators from SPY (or any financial instrument) for hybrid quantitative + NLP modeling.

4. **Proven in Literature & Industry**  
   This FinBERT approach is widely used in:
   - Hedge fund sentiment models
   - Academic market prediction research
   - Financial risk management applications

## Why I Could Not Run It Natively
While this is the best approach, my current setup has limitations:
- My **2018 MacBook** cannot run PyTorch ≥ 2.6 due to lack of support for certain macOS builds.
- Hugging Face **blocked `.bin` weight loading** for security (CVE-2025-32434), requiring either:
  - A `safetensors` model format, or
  - Upgrading to PyTorch ≥ 2.6
- The **UTM virtualized Linux environment** that would allow PyTorch ≥ 2.6 runs extremely slowly on my machine and is impractical for iterative development.

## Intended Usage (If No Hardware/OS Limitations)
If running on a supported Linux system or modern macOS:
1. Install dependencies:
   ```bash
   pip install torch==2.6.0 transformers pandas
2. Download financial news headlines into news_headlines.csv (with a headline column).

3. Run:
  ```bash
   python finbert_sentiment_pipeline.py

4. Use news_with_sentiment.csv as input to your predictive modeling pipeline (e.g., XGBoost, LSTM).

## Recommended Next Steps
Merge daily aggregated sentiment scores into SPY technical indicator datasets.
Train hybrid models combining technical + sentiment features.
Experiment with transformer-based late fusion models for improved accuracy.




