"""Download and combine Bitcoin news datasets from multiple sources for ML training."""

import json
import kagglehub
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_existing_json_data() -> pd.DataFrame:
    """Load existing data.json file if it exists."""
    json_path = Path("data.json")
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract relevant fields for Bitcoin
        bitcoin_news = []
        for article in data:
            if 'coin' in article and 'BTC' in article.get('coin', []):
                bitcoin_news.append({
                    'date': article.get('pubDate', ''),
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('link', ''),
                    'source': article.get('source_name', ''),
                    'sentiment': article.get('sentiment', ''),
                    'sentiment_positive': article.get('sentiment_stats', {}).get('positive', None),
                    'sentiment_neutral': article.get('sentiment_stats', {}).get('neutral', None),
                    'sentiment_negative': article.get('sentiment_stats', {}).get('negative', None)
                })

        print(f"Loaded {len(bitcoin_news)} Bitcoin articles from data.json")
        return pd.DataFrame(bitcoin_news)
    return pd.DataFrame()

def download_kaggle_dataset(dataset_name: str) -> str:
    """Download a Kaggle dataset and return the path."""
    print(f"Downloading {dataset_name}...")
    path = kagglehub.dataset_download(dataset_name)
    print(f"Dataset downloaded to: {path}")
    return path

def load_bitcoin_news_dataset() -> pd.DataFrame:
    """Download and load Bitcoin-News Dataset."""
    try:
        path = download_kaggle_dataset("ashirwadsangwan/bitcoinnews-dataset")
        csv_files = list(Path(path).glob("*.csv"))

        if csv_files:
            df = pd.read_csv(csv_files[0])
            print(f"Loaded {len(df)} rows from Bitcoin-News Dataset")
            # Standardize column names
            df = df.rename(columns={
                'Date': 'date',
                'Title': 'title',
                'Description': 'description',
                'URL': 'url',
                'Source': 'source'
            })
            return df
    except Exception as e:
        print(f"Error loading Bitcoin-News Dataset: {e}")
    return pd.DataFrame()

def load_crypto_news_plus() -> pd.DataFrame:
    """Download and load Crypto News + Dataset."""
    try:
        path = download_kaggle_dataset("oliviervha/crypto-news")
        csv_files = list(Path(path).glob("*.csv"))

        if csv_files:
            df = pd.read_csv(csv_files[0])
            # Filter for Bitcoin-related news
            bitcoin_mask = df['title'].str.contains('bitcoin|btc', case=False, na=False) | \
                          df['text'].str.contains('bitcoin|btc', case=False, na=False)
            df = df[bitcoin_mask]
            print(f"Loaded {len(df)} Bitcoin-related rows from Crypto News +")
            # Standardize column names
            df = df.rename(columns={
                'text': 'content',
                'sentiment': 'sentiment'
            })
            return df
    except Exception as e:
        print(f"Error loading Crypto News +: {e}")
    return pd.DataFrame()

def load_crypto_news_with_prices() -> pd.DataFrame:
    """Download and load Crypto News Headlines & Market Prices Dataset."""
    try:
        path = download_kaggle_dataset("aaroncbastian/crypto-news-headlines-and-market-prices-by-date")
        csv_files = list(Path(path).glob("*.csv"))

        if csv_files:
            df = pd.read_csv(csv_files[0])
            # Filter for Bitcoin
            bitcoin_df = df[df['currency'] == 'BTC'] if 'currency' in df.columns else df
            print(f"Loaded {len(bitcoin_df)} Bitcoin rows from Crypto News with Prices")
            # Standardize column names
            bitcoin_df = bitcoin_df.rename(columns={
                'headline': 'title',
                'price': 'btc_price',
                'open': 'btc_open',
                'close': 'btc_close',
                'high': 'btc_high',
                'low': 'btc_low',
                'volume': 'btc_volume'
            })
            return bitcoin_df
    except Exception as e:
        print(f"Error loading Crypto News with Prices: {e}")
    return pd.DataFrame()

def load_historical_crypto_news() -> pd.DataFrame:
    """Download and load News about major cryptocurrencies 2013-2018."""
    try:
        path = download_kaggle_dataset("kashnitsky/news-about-major-cryptocurrencies-20132018-40k")
        csv_files = list(Path(path).glob("*.csv"))

        if csv_files:
            df = pd.read_csv(csv_files[0])
            # Filter for Bitcoin
            bitcoin_df = df[df['currency'] == 'Bitcoin'] if 'currency' in df.columns else df
            print(f"Loaded {len(bitcoin_df)} Bitcoin rows from Historical Crypto News (2013-2018)")
            # Standardize column names
            bitcoin_df = bitcoin_df.rename(columns={
                'text': 'content',
                'created_at': 'date'
            })
            return bitcoin_df
    except Exception as e:
        print(f"Error loading Historical Crypto News: {e}")
    return pd.DataFrame()

def combine_and_save_datasets():
    """Combine all datasets and save as a single CSV."""
    print("\n" + "="*80)
    print("DOWNLOADING AND COMBINING BITCOIN NEWS DATASETS")
    print("="*80 + "\n")

    # Load all datasets
    datasets = []

    # Existing JSON data
    json_df = load_existing_json_data()
    if not json_df.empty:
        datasets.append(json_df)

    # Kaggle datasets
    datasets.append(load_bitcoin_news_dataset())
    datasets.append(load_crypto_news_plus())
    datasets.append(load_crypto_news_with_prices())
    datasets.append(load_historical_crypto_news())

    # Remove empty dataframes
    datasets = [df for df in datasets if not df.empty]

    if not datasets:
        print("\nNo datasets were successfully loaded!")
        return

    # Combine all datasets
    print("\n" + "-"*80)
    print("Combining datasets...")

    # Find common columns across all datasets
    all_columns = set()
    for df in datasets:
        all_columns.update(df.columns)

    # Create a master dataframe with all possible columns
    combined_df = pd.concat(datasets, ignore_index=True, sort=False)

    # Clean and standardize dates
    if 'date' in combined_df.columns:
        combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
        combined_df = combined_df.sort_values('date', ascending=False)

    # Remove duplicates based on title
    if 'title' in combined_df.columns:
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['title'], keep='first')
        print(f"Removed {initial_count - len(combined_df)} duplicate articles")

    # Save to CSV
    output_file = "bitcoin_news_combined.csv"
    combined_df.to_csv(output_file, index=False, encoding='utf-8')

    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    print(f"Total articles: {len(combined_df)}")
    print(f"Columns: {list(combined_df.columns)}")
    print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    print(f"\nOutput saved to: {output_file}")
    print("="*80 + "\n")

    # Show sample
    print("Sample of the data:")
    print(combined_df[['date', 'title']].head(10))

    return combined_df

if __name__ == "__main__":
    combined_df = combine_and_save_datasets()
