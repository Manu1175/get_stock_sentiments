import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Ensure output directory exists
save_dir = "./plots/compare"
os.makedirs(save_dir, exist_ok=True)

# Load CSVs
df_vader = pd.read_csv("data/reddit_trending_sentiment.csv")
df_bert = pd.read_csv("data/reddit_trending_sentiment_transformers.csv")

# ---------------------------
# 1️⃣ Overall sentiment distribution
# ---------------------------
def sentiment_distribution(df, model_name):
    df['dominant'] = df[['bullish','bearish','neutral']].idxmax(axis=1)
    counts = df['dominant'].value_counts(normalize=True)
    counts = counts.reindex(['bullish','bearish','neutral'], fill_value=0)
    return counts.rename(model_name)

vader_dist = sentiment_distribution(df_vader, 'Vader+TFIDF')
bert_dist = sentiment_distribution(df_bert, 'Transformers')
dist_df = pd.concat([vader_dist, bert_dist], axis=1)

# Bar plot
plt.figure(figsize=(8,6))
dist_df.plot(kind='bar')
plt.ylabel('Proportion')
plt.title('Dominant Sentiment Distribution: Vader vs Transformers')
plt.xticks(rotation=0)
bar_path = os.path.join(save_dir, f"dominant_sentiment_bar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
plt.savefig(bar_path, dpi=300, bbox_inches='tight')
plt.show()

# Difference heatmap
plt.figure(figsize=(6,2))
diff = dist_df['Transformers'] - dist_df['Vader+TFIDF']
sns.heatmap(diff.to_frame().T, annot=True, cmap='RdBu', center=0, cbar_kws={'label': 'Difference (Transformers - Vader)'})
plt.title('Sentiment Assignment Differences')
heatmap_path = os.path.join(save_dir, f"sentiment_difference_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------
# 2️⃣ Ticker-wise sentiment scatter plot with company names
# ---------------------------
def ticker_sentiment_proportion(df):
    df['dominant'] = df[['bullish','bearish','neutral']].idxmax(axis=1)
    prop_df = df.groupby('ticker')['dominant'].value_counts(normalize=True).unstack(fill_value=0)
    for col in ['bullish','bearish','neutral']:
        if col not in prop_df.columns:
            prop_df[col] = 0
    # Add company names
    ticker_to_company = df.groupby('ticker')['company_name'].first()
    prop_df['company_name'] = ticker_to_company
    # Exclude unknown or empty
    prop_df = prop_df[prop_df['company_name'].notnull() & (prop_df['company_name'] != 'Unknown') & (prop_df['company_name'] != '')]
    return prop_df[['bullish','bearish','neutral','company_name']]

vader_prop = ticker_sentiment_proportion(df_vader)
bert_prop = ticker_sentiment_proportion(df_bert)

# Rename company_name columns to avoid join conflicts
vader_prop = vader_prop.rename(columns={'company_name': 'company_name_vader'})
bert_prop = bert_prop.rename(columns={'company_name': 'company_name_bert'})

# Merge on tickers
merged = vader_prop.join(bert_prop, lsuffix='_vader', rsuffix='_bert', how='inner')
merged['company_name'] = merged['company_name_vader']

# Scatter plot per sentiment
plt.figure(figsize=(16,8))
colors = {'bullish':'green','bearish':'red','neutral':'grey'}

for sentiment in ['bullish','bearish','neutral']:
    plt.scatter(
        merged[f'{sentiment}_vader'], 
        merged[f'{sentiment}_bert'], 
        label=sentiment, s=100, alpha=0.7, color=colors[sentiment]
    )

# Diagonal line
plt.plot([0,1],[0,1],'k--', alpha=0.5)

# Annotate points with TICKER only
threshold = 0.05  # optional: label only near diagonal
for idx, row in merged.iterrows():
    for sentiment in ['bullish','bearish','neutral']:
        diff = abs(row[f'{sentiment}_vader'] - row[f'{sentiment}_bert'])
        if diff < threshold:  # only label agreements
            plt.text(
                row[f'{sentiment}_vader'] + 0.01, 
                row[f'{sentiment}_bert'] + 0.01, 
                idx,  # ticker instead of company name
                fontsize=8,
                alpha=0.9
            )

# Create a separate legend mapping tickers to company names
ticker_to_company = merged['company_name'].to_dict()
legend_text = "\n".join([f"{ticker}: {name}" for ticker, name in ticker_to_company.items()])

plt.xlabel('Vader+TFIDF Dominant Sentiment Proportion')
plt.ylabel('Transformers Dominant Sentiment Proportion')
plt.title('Ticker-wise Dominant Sentiment: Vader vs Transformers (Tickers on plot)')
plt.legend(title='Sentiment', loc='upper left')
plt.grid(True)

scatter_path = os.path.join(
    save_dir, 
    f"ticker_sentiment_scatter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
)
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
plt.show()

# Print ticker → company mapping for reference
print("Ticker → Company mapping:\n", legend_text)

print(f"Plots saved in {save_dir}")
