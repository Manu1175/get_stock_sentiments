import praw
import re
import pandas as pd
from collections import Counter
from datetime import datetime
import time
import os
import yfinance as yf
from nltk.corpus import stopwords
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import numpy as np

# -----------------------------
# 1️⃣ Hugging Face Sentiment Model
# -----------------------------
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def compute_transformer_sentiment(posts, min_length=5, neutral_threshold=0.65):
    """
    Compute sentiment scores using Hugging Face Transformers (DistilBERT SST-2).
    """
    scores = {'bullish': 0, 'bearish': 0, 'neutral': 0}
    filtered_posts = [p for p in posts if p and len(p.strip()) >= min_length]
    if not filtered_posts:
        return scores

    results = sentiment_model(filtered_posts, truncation=True)
    for res in results:
        label, conf = res['label'], res['score']
        if conf < neutral_threshold:
            scores['neutral'] += 1
        elif label == "POSITIVE":
            scores['bullish'] += 1
        elif label == "NEGATIVE":
            scores['bearish'] += 1

    total = len(filtered_posts)
    return {k: v / total for k, v in scores.items()}


# -----------------------------
# 2️⃣ Reddit Setup
# -----------------------------
reddit = praw.Reddit(
    client_id="",
    client_secret="",
    user_agent="stock_sentiment_script"
)

subreddits = ['stocks', 'wallstreetbets', 'investing', 'StockMarket', 'options']
top_limit = 500
trending_limit = 300
update_interval = 1800  # 30 minutes
stop_words = set(stopwords.words('english'))

# -----------------------------
# 3️⃣ Helper Functions
# -----------------------------
def get_company_name(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get("longName", "Unknown")
    except Exception:
        return "Unknown"

def fetch_posts():
    posts = []
    for sub in subreddits:
        subreddit = reddit.subreddit(sub)
        for post in subreddit.hot(limit=top_limit):
            posts.append(post.title + " " + post.selftext)
    return posts

def extract_tickers(posts):
    tickers = []
    for post in posts:
        tickers += re.findall(r"\$[A-Z]{1,5}", post)
    return [t.replace("$", "") for t in tickers]

def group_posts_by_ticker(posts, trending_tickers):
    grouped = {ticker: [] for ticker in trending_tickers}
    for post in posts:
        for ticker in trending_tickers:
            if f"${ticker}" in post:
                grouped[ticker].append(post)
    return grouped

def top_keywords(posts, n=5):
    words = []
    for post in posts:
        words += [w.lower() for w in re.findall(r'\b\w+\b', post) if w.lower() not in stop_words]
    counter = Counter(words)
    return [word for word, _ in counter.most_common(n)]

def compute_sentiment_for_trending(posts, trending_tickers):
    total_posts = len(posts)
    grouped = group_posts_by_ticker(posts, trending_tickers)
    data = []

    for ticker, ticker_posts in grouped.items():
        if ticker_posts:
            sentiment_scores = compute_transformer_sentiment(ticker_posts)
            bullish = sentiment_scores.get('bullish', 0)
            bearish = sentiment_scores.get('bearish', 0)
            neutral = sentiment_scores.get('neutral', 0)
            sentiment_ratio = bullish / (bearish + 1)
            engagement_score = len(ticker_posts) / total_posts
            keywords = ', '.join(top_keywords(ticker_posts, 5))

            sentiment = {
                'ticker': ticker,
                'company_name': get_company_name(ticker),
                'bullish': bullish,
                'bearish': bearish,
                'neutral': neutral,
                'sentiment_ratio': sentiment_ratio,
                'num_posts': len(ticker_posts),
                'engagement_score': engagement_score,
                'top_keywords': keywords,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            data.append(sentiment)

    return pd.DataFrame(data)


# -----------------------------
# 4️⃣ Visualization
# -----------------------------
def plot_sentiment_combined(df, top_n=10, save_dir=None):
    if df.empty:
        print("No data available to plot.")
        return

    df = df[df['company_name'] != 'Unknown']
    if df.empty:
        print("No valid company names to plot.")
        return

    latest_data = df.sort_values('timestamp').groupby('ticker').tail(1)
    top10 = latest_data.sort_values('bullish', ascending=False).head(top_n)
    if top10.empty:
        print("No top 10 data available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.set_style("whitegrid")

    # --- Left subplot: normalized stacked bars ---
    ax = axes[0]
    indices = range(len(top10))
    base_colors = {'bullish': 'green', 'bearish': 'red', 'neutral': 'grey'}

    for i, row in enumerate(top10.itertuples()):
        total = row.bullish + row.bearish + row.neutral
        normalized = {'bullish': row.bullish / total,
                      'bearish': row.bearish / total,
                      'neutral': row.neutral / total} if total > 0 else {'bullish':0,'bearish':0,'neutral':0}

        bottom = 0
        for key in ['bullish','bearish','neutral']:
            value = normalized[key]
            if value > 0:
                ax.bar(i, value, bottom=bottom, color=base_colors[key])
                ax.text(i, bottom + value / 2, f"{value:.2f}",
                        ha='center', va='center', fontsize=9, color='black')
                bottom += value

    x_labels = [f"{row['company_name']} ({row['ticker']})" for _, row in top10.iterrows()]
    ax.set_xticks(indices)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel('Normalized Sentiment Score (0-1)')
    ax.set_xlabel('Company (Ticker)')
    ax.set_title('Top Tickers Sentiment (Normalized)')

    bull_patch = mpatches.Patch(color='green', label='Bullish')
    bear_patch = mpatches.Patch(color='red', label='Bearish')
    neut_patch = mpatches.Patch(color='grey', label='Neutral')
    ax.legend(handles=[bull_patch, bear_patch, neut_patch])

    # --- Right subplot: bubble chart ---
    ax2 = axes[1]

    # Bubble sizes
    sizes = (top10['num_posts'] / top10['num_posts'].max()) * 2000

    # Bubble colors by dominant sentiment
    bubble_colors = []
    for _, row in top10.iterrows():
        if row['bullish'] >= row['bearish'] and row['bullish'] >= row['neutral']:
            bubble_colors.append('green')
        elif row['bearish'] >= row['bullish'] and row['bearish'] >= row['neutral']:
            bubble_colors.append('red')
        else:
            bubble_colors.append('grey')

    scatter = ax2.scatter(
        top10['engagement_score'],
        top10['sentiment_ratio'],
        s=sizes,
        c=bubble_colors,
        alpha=0.6,
        edgecolors='w',
        linewidth=0.5
    )

    # Add labels
    for i, row in top10.iterrows():
        ax2.text(row['engagement_score'], row['sentiment_ratio'] + 0.02,
                 f"{row['ticker']}", ha='center', va='bottom', fontsize=9)

    ax2.set_xlabel('Engagement Score')
    ax2.set_ylabel('Sentiment Ratio (Bullish / (Bearish + 1))')
    ax2.set_title('Sentiment Ratio vs Engagement Score (Bubble Size = #Posts)')

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"top10_sentiment_transformers_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {save_path}")

    plt.show(block=False)
    plt.pause(1)
    plt.clf()


# -----------------------------
# 5️⃣ Main Loop
# -----------------------------
if __name__ == "__main__":
    all_data = pd.DataFrame()
    
    while True:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching posts...")
        posts = fetch_posts()
        
        tickers_counter = Counter(extract_tickers(posts))
        trending_tickers = [ticker for ticker, _ in tickers_counter.most_common(trending_limit)]
        
        print(f"Trending tickers: {trending_tickers}")
        
        df = compute_sentiment_for_trending(posts, trending_tickers)
        all_data = pd.concat([all_data, df], ignore_index=True)
        
        csv_filename = "data/reddit_trending_sentiment_transformers.csv"
        all_data.to_csv(
            csv_filename,
            mode="a",
            header=not os.path.exists(csv_filename),
            index=False
        )
        
        print(df)
        print(f"Data saved to {csv_filename}. Next update in {update_interval/60} minutes.\n")
        
        plot_sentiment_combined(df, top_n=10, save_dir='plots')
        
        time.sleep(update_interval)

