from newsapi import NewsApiClient
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Initialize
newsapi = NewsApiClient(api_key= NEWS_API_KEY)  # Replace with your actual News API key

# Set date range
def fetch_news_for_date(date_str):
    everything = newsapi.get_everything(
        q='S&P 500 OR SPY OR market',
        from_param=date_str,
        to=date_str,
        language='en',
        sort_by='relevancy',
        page_size=100
    )
    articles = everything['articles']
    return [(a['publishedAt'][:10], a['title']) for a in articles]

# Example for past 30 days
today = datetime.today()
dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]

all_news = []
for date_str in dates:
    try:
        daily_news = fetch_news_for_date(date_str)
        all_news.extend(daily_news)
    except Exception as e:
        print(f"Failed on {date_str}: {e}")

df_news = pd.DataFrame(all_news, columns=['date', 'headline'])
df_news.to_csv("news_headlines.csv", index=False)
