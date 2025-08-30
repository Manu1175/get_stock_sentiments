import praw
import re
import pandas as pd
from collections import Counter
from datetime import datetime
import time
from nltk.corpus import stopwords
from nlp_utils import compute_tfidf_sentiment
import os
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def compute_vader_sentiment(posts, min_length=5):
    """
    Compute VADER sentiment scores for a list of text posts with basic filtering.
    
    Parameters:
        posts (list of str): List of Reddit post contents (title + body).
        min_length (int): Minimum number of characters a post must have to be considered.
        
    Returns:
        dict: Normalized sentiment scores with keys:
            - 'bullish': proportion of positive posts
            - 'bearish': proportion of negative posts
            - 'neutral': proportion of neutral posts
    """
    scores = {'bullish': 0, 'bearish': 0, 'neutral': 0}
    
    # Filter out empty or very short posts to avoid noise
    filtered_posts = [p for p in posts if p and len(p.strip()) >= min_length]
    
    for post in filtered_posts:
        vs = analyzer.polarity_scores(post)  # returns 'neg', 'neu', 'pos', 'compound'
        compound = vs['compound']  # range -1 to 1
        if compound >= 0.05:
            scores['bullish'] += 1
        elif compound <= -0.05:
            scores['bearish'] += 1
        else:
            scores['neutral'] += 1
    
    total = len(filtered_posts)
    if total > 0:
        # Normalize to proportions
        scores = {k: v / total for k, v in scores.items()}
    else:
        # Avoid division by zero
        scores = {'bullish': 0, 'bearish': 0, 'neutral': 0}
    
    return scores


def compute_sentiment_for_trending(posts_dict, trending_tickers):
    """
    Compute sentiment scores for multiple trending tickers using TF-IDF sentiment.

    Parameters:
        posts_dict (dict): Mapping of ticker symbol -> list of Reddit posts.
        trending_tickers (list of str): List of trending ticker symbols.

    Returns:
        list of dict: Each dictionary contains sentiment scores for a ticker.
    """
    results = []

    for ticker in trending_tickers:
        ticker_posts = posts_dict.get(ticker, [])
        sentiment_scores = compute_tfidf_sentiment(ticker_posts)
        sentiment_scores["ticker"] = ticker
        results.append(sentiment_scores)

    return results

# --- Reddit setup ---
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
# 2️⃣ Helper Functions
# -----------------------------
# 🔹 Helper: map ticker to company full name
def get_company_name(ticker):
    """
    Retrieve the full company name for a stock ticker using yfinance.
    
    Parameters:
        ticker (str): Stock ticker symbol.
        
    Returns:
        str: Full company name or "Unknown" if unavailable.
    """
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get("longName", "Unknown")
    except Exception:
        return "Unknown"

def fetch_posts():
    """
    Fetch posts from predefined subreddits using PRAW.
    
    Returns:
        list of str: Concatenated post titles and selftexts.
    """
    posts = []
    for sub in subreddits:
        subreddit = reddit.subreddit(sub)
        for post in subreddit.hot(limit=top_limit):
            posts.append(post.title + " " + post.selftext)
    return posts

def extract_tickers(posts):
    """
    Extract stock tickers from a list of posts using regex.
    
    Parameters:
        posts (list of str): Reddit post contents.
        
    Returns:
        list of str: List of tickers without the '$' symbol.
    """
    tickers = []
    for post in posts:
        tickers += re.findall(r"\$[A-Z]{1,5}", post)
    return [t.replace("$", "") for t in tickers]

def group_posts_by_ticker(posts, trending_tickers):
    """
    Group posts by ticker symbol.
    
    Parameters:
        posts (list of str): List of Reddit posts.
        trending_tickers (list of str): List of trending ticker symbols.
        
    Returns:
        dict: Mapping ticker -> list of posts mentioning that ticker.
    """
    grouped = {ticker: [] for ticker in trending_tickers}
    for post in posts:
        for ticker in trending_tickers:
            if f"${ticker}" in post:
                grouped[ticker].append(post)
    return grouped

def top_keywords(posts, n=5):
    """
    Extract the top n keywords from a list of posts, excluding stopwords.
    
    Parameters:
        posts (list of str): Reddit post contents.
        n (int): Number of top keywords to return.
        
    Returns:
        list of str: Top n most frequent keywords.
    """
    words = []
    for post in posts:
        words += [w.lower() for w in re.findall(r'\b\w+\b', post) if w.lower() not in stop_words]
    counter = Counter(words)
    return [word for word, _ in counter.most_common(n)]

def compute_combined_sentiment(posts, tfidf_weight=0.5, vader_weight=0.5):
    """
    Compute combined sentiment using both TF-IDF and VADER scores.

    Parameters:
        posts (list of str): Reddit post contents.
        tfidf_weight (float): Weight for TF-IDF sentiment.
        vader_weight (float): Weight for VADER sentiment.

    Returns:
        dict: Combined sentiment scores with keys 'bullish', 'bearish', 'neutral'.
    """
    if not posts:
        return {'bullish': 0, 'bearish': 0, 'neutral': 0}

    # --- VADER ---
    vader_scores = compute_vader_sentiment(posts)

    # --- TF-IDF ---
    tfidf_scores = compute_tfidf_sentiment(posts)

    # --- Combine ---
    combined = {}
    for key in ['bullish', 'bearish', 'neutral']:
        combined[key] = tfidf_scores.get(key, 0) * tfidf_weight + vader_scores.get(key, 0) * vader_weight

    return combined


def compute_sentiment_for_trending(posts, trending_tickers):
    """
    Compute sentiment metrics for trending tickers using a hybrid TF-IDF + VADER approach.

    Parameters:
        posts (list of str): List of Reddit posts.
        trending_tickers (list of str): Trending tickers to analyze.

    Returns:
        pd.DataFrame: DataFrame containing sentiment, engagement, top keywords, and timestamp.
    """
    total_posts = len(posts)
    grouped = group_posts_by_ticker(posts, trending_tickers)
    data = []

    for ticker, ticker_posts in grouped.items():
        if ticker_posts:
            sentiment_scores = compute_combined_sentiment(ticker_posts, tfidf_weight=0.5, vader_weight=0.5)
            bullish = sentiment_scores.get('bullish', 0)
            bearish = sentiment_scores.get('bearish', 0)
            neutral = sentiment_scores.get('neutral', 0)
            sentiment_ratio = bullish / (bearish + 1)  # avoid division by zero
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


import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 4️⃣ Visualization Function
# -----------------------------
def plot_sentiment_combined(df, top_n=10, save_dir=None):
    """
    Plot top N tickers with two subplots:
    1. Normalized stacked sentiment bars (0-1).
    2. Sentiment ratio vs engagement score scatter plot.
    Companies with 'Unknown' name are excluded.
    Optionally saves the figure as an image with datestamp in filename.

    Parameters:
        df (pd.DataFrame): DataFrame with sentiment metrics including
                           'ticker', 'company_name', 'bullish', 'bearish', 'neutral',
                           'sentiment_ratio', 'engagement_score', 'timestamp'.
        top_n (int): Number of tickers to show.
        save_dir (str, optional): Directory to save the image. If None, image is not saved.
    """
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

    # Left subplot: normalized stacked bars
    ax = axes[0]
    bar_width = 0.5
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
                ax.text(i, bottom + value / 2, f"{value:.2f}", ha='center', va='center', fontsize=9, color='black')
                bottom += value

    x_labels = [f"{row['company_name']} ({row['ticker']})" for _, row in top10.iterrows()]
    ax.set_xticks(indices)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel('Normalized Sentiment Score (0-1)')
    ax.set_xlabel('Company (Ticker)')
    ax.set_title('Top Tickers Sentiment (Normalized)')
    ax.legend(['Bullish', 'Bearish', 'Neutral'])

    # Right subplot: sentiment ratio vs engagement
    ax2 = axes[1]
    for i, row in top10.iterrows():
        ax2.scatter(row['engagement_score'], row['sentiment_ratio'], s=100, color='blue')
        ax2.text(row['engagement_score'], row['sentiment_ratio'] + 0.02,
                 f"{row['ticker']}", ha='center', va='bottom', fontsize=9)

    ax2.set_xlabel('Engagement Score')
    ax2.set_ylabel('Sentiment Ratio (Bullish / (Bearish + 1))')
    ax2.set_title('Sentiment Ratio vs Engagement Score')

    plt.tight_layout()

    # Save figure if directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"top10_sentiment_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {save_path}")

    plt.show(block=False)
    plt.pause(1)
    plt.clf()
    
    
# -----------------------------
# 3️⃣ Main Loop
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
        
        csv_filename = f"data/reddit_trending_sentiment.csv"
        # Append without overwriting
        all_data.to_csv(
            csv_filename,
            mode="a",          # append instead of overwrite
            header=not os.path.exists(csv_filename),  # write header only if file doesn't exist
            index=False
        )
        
        print(df)
        print(f"Data saved to {csv_filename}. Next update in {update_interval/60} minutes.\n")
        
        # -----------------------------
        # Show the plot using existing function
        # -----------------------------
        plot_sentiment_combined(df, top_n=10, save_dir='plots')
        
        time.sleep(update_interval)
        