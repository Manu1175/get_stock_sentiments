# Reddit Stock Sentiment Analysis

This project collects Reddit posts from finance-related subreddits and computes sentiment scores for trending stock tickers using multiple NLP techniques. It also visualizes sentiment trends and engagement metrics for top tickers over time.

---

## Features

1. **Reddit Data Collection**
   - Fetches posts from subreddits: `stocks`, `wallstreetbets`, `investing`, `StockMarket`, `options`.
   - Collects titles and selftexts from top `hot` posts.
   - Extracts trending stock tickers mentioned in posts using regex.

2. **Sentiment Analysis**
   - **TF-IDF + Keyword-based sentiment**
     - Uses `nltk` for tokenization and preprocessing.
     - Counts bullish/bearish/neutral keywords.
   - **VADER Sentiment**
     - Computes sentiment using VADER (`vaderSentiment`).
     - Normalizes scores to proportions.
   - **Transformer-based Sentiment**
     - Hugging Face `distilbert-base-uncased-finetuned-sst-2-english`.
     - Provides fine-grained sentiment classification (`POSITIVE`, `NEGATIVE`, `NEUTRAL`).

3. **Data Processing**
   - Groups posts by ticker.
   - Computes combined sentiment scores (TF-IDF + VADER).
   - Extracts top keywords per ticker.
   - Calculates engagement metrics (proportion of posts mentioning a ticker).

4. **Visualization**
   - Normalized stacked bar plots for sentiment.
   - Scatter/bubble plots: sentiment ratio vs engagement score.
   - Saves plots with timestamps for historical tracking.

5. **Continuous Monitoring**
   - Runs in a loop with configurable update intervals (default 30 minutes).
   - Appends new sentiment data to CSV files without overwriting existing data.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/reddit-stock-sentiment.git
cd reddit-stock-sentiment
```

2. Install required Python packages:

```bash
pip install -r requirements.txt
```

3. NLTK downloads:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```
4. Optional: For transformer-based sentiment analysis:

```bash
pip install transformers torch
```

## Usage
### Running TF-IDF + VADER Sentiment

```bash
python reddit_sentiment.py
```
This script:

* Fetches posts.
* Computes sentiment for trending tickers.
* Saves results to data/reddit_trending_sentiment.csv.
* Displays and saves plots to plots/.

### Running Transformer-based Sentiment

```bash
python reddit_sentiment_transformers.py
```

This script:

* Uses Hugging Face Transformers for sentiment analysis.
* Saves results to data/reddit_trending_sentiment_transformers.csv.
* Visualizes sentiment with bubble plots.
* Configuration
* subreddits: List of subreddits to fetch posts from.
* top_limit: Maximum number of posts per subreddit.
* trending_limit: Number of top trending tickers to track.
* update_interval: Time between fetch cycles (seconds, default 1800 = 30 minutes).
* CSV filenames and plot directories are configurable in the scripts.

### File Structure

```graphql
.
├── reddit_sentiment.py                 # Main loop: TF-IDF + VADER sentiment
├── reddit_sentiment_transformers.py    # Main loop: Transformer sentiment
├── nlp_utils.py                        # TF-IDF + keyword-based sentiment functions
├── data/                               # CSV output storage
├── plots/                              # Generated plots
├── requirements.txt                     # Dependencies
└── README.md
```

### Dependencies

* praw – Reddit API wrapper

* nltk – Tokenization and stopwords

* sklearn – TF-IDF vectorizer

* vaderSentiment – VADER sentiment analysis

* yfinance – Retrieve company names

* pandas – Data manipulation

* matplotlib & seaborn – Visualization

* transformers & torch – Transformer-based sentiment

### Notes

Ensure you have valid Reddit API credentials (client_id, client_secret, user_agent).

Large-scale scraping may hit Reddit API rate limits; consider caching or limiting frequency.

CSV and plot directories are created automatically if missing.