# nlp_utils.py
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')  # make sure tokenizer is available

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]  # keep only words
    return " ".join(tokens)

def compute_tfidf_sentiment(posts):
    """
    Input: list of text posts
    Output: dict with bullish/bearish/neutral counts
    """
    if not posts:
        return {"bullish": 0, "bearish": 0, "neutral": 0}

    # Simple keyword-based sentiment for demonstration
    bullish_keywords = ["buy", "bull", "long", "up"]
    bearish_keywords = ["sell", "bear", "short", "down"]

    processed_texts = [preprocess_text(p) for p in posts]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_texts)

    scores = {"bullish": 0, "bearish": 0, "neutral": 0}

    for text in processed_texts:
        text_tokens = text.split()
        bullish_count = sum(1 for t in text_tokens if t in bullish_keywords)
        bearish_count = sum(1 for t in text_tokens if t in bearish_keywords)
        if bullish_count > bearish_count:
            scores["bullish"] += 1
        elif bearish_count > bullish_count:
            scores["bearish"] += 1
        else:
            scores["neutral"] += 1

    return scores