"""Create final Bitcoin news dataset with price data for ML training."""

import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path

def fetch_bitcoin_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical Bitcoin prices from CoinGecko API (free, no API key).

    Args:
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'

    Returns:
        DataFrame with Bitcoin prices
    """
    print("Fetching Bitcoin price data from CoinGecko...")

    # Convert dates to timestamps
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    params = {
        'vs_currency': 'usd',
        'from': start_ts,
        'to': end_ts
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Extract prices
        prices = data.get('prices', [])
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['btc_price'] = df['price']
        df = df[['date', 'btc_price']]

        print(f"Fetched {len(df)} price points from {df['date'].min()} to {df['date'].max()}")
        return df

    except Exception as e:
        print(f"Error fetching Bitcoin prices: {e}")
        return pd.DataFrame()

def merge_news_with_prices(news_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge news articles with Bitcoin prices by date.

    Args:
        news_df: DataFrame with news articles
        prices_df: DataFrame with Bitcoin prices

    Returns:
        Merged DataFrame
    """
    # Ensure date columns are datetime
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
    prices_df['date'] = pd.to_datetime(prices_df['date'], errors='coerce')

    # Round dates to nearest day for matching
    news_df['date_day'] = news_df['date'].dt.date
    prices_df['date_day'] = prices_df['date'].dt.date

    # Get daily average price
    daily_prices = prices_df.groupby('date_day')['btc_price'].agg(['mean', 'min', 'max']).reset_index()
    daily_prices.columns = ['date_day', 'btc_price_avg', 'btc_price_min', 'btc_price_max']

    # Merge news with prices
    merged_df = news_df.merge(daily_prices, on='date_day', how='left')

    # Calculate price change (next day vs current day) for target variable
    # This will be what we want to predict
    daily_prices_sorted = daily_prices.sort_values('date_day')
    daily_prices_sorted['btc_price_next_day'] = daily_prices_sorted['btc_price_avg'].shift(-1)
    daily_prices_sorted['btc_price_change'] = daily_prices_sorted['btc_price_next_day'] - daily_prices_sorted['btc_price_avg']
    daily_prices_sorted['btc_price_change_pct'] = (daily_prices_sorted['btc_price_change'] / daily_prices_sorted['btc_price_avg']) * 100

    # Merge with price changes
    merged_df = merged_df.merge(
        daily_prices_sorted[['date_day', 'btc_price_next_day', 'btc_price_change', 'btc_price_change_pct']],
        on='date_day',
        how='left'
    )

    # Create binary target: 1 if price goes up, 0 if down
    merged_df['btc_price_up'] = (merged_df['btc_price_change'] > 0).astype(int)

    # Drop temporary column
    merged_df = merged_df.drop('date_day', axis=1)

    return merged_df

def create_final_dataset():
    """Create the final Bitcoin news dataset with prices."""
    print("\n" + "="*80)
    print("CREATING FINAL BITCOIN NEWS DATASET FOR ML TRAINING")
    print("="*80 + "\n")

    # Load all news data
    news_files = [
        'bitcoin_news_combined.csv',
        'bitcoin_news_recent_scraped.csv'
    ]

    all_news = []
    for file in news_files:
        if Path(file).exists():
            print(f"Loading {file}...")
            df = pd.read_csv(file)
            all_news.append(df)
            print(f"  Loaded {len(df)} articles")

    if not all_news:
        print("No news data found! Run download_bitcoin_news.py first.")
        return

    # Combine all news
    news_df = pd.concat(all_news, ignore_index=True)
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
    news_df = news_df.dropna(subset=['date'])

    # Remove duplicates
    initial_count = len(news_df)
    news_df = news_df.drop_duplicates(subset=['title'], keep='first')
    print(f"\nRemoved {initial_count - len(news_df)} duplicates")
    print(f"Total unique articles: {len(news_df)}")

    # Get date range
    min_date = news_df['date'].min().strftime('%Y-%m-%d')
    max_date = news_df['date'].max().strftime('%Y-%m-%d')
    print(f"Date range: {min_date} to {max_date}")

    # Fetch Bitcoin prices
    prices_df = fetch_bitcoin_prices(min_date, max_date)

    if prices_df.empty:
        print("\nWarning: Could not fetch Bitcoin prices. Saving news data only.")
        final_df = news_df
    else:
        # Merge news with prices
        print("\nMerging news with Bitcoin prices...")
        final_df = merge_news_with_prices(news_df, prices_df)

        # Show price statistics
        print(f"\nPrice statistics:")
        print(f"  Articles with price data: {final_df['btc_price_avg'].notna().sum()}")
        print(f"  Average BTC price: ${final_df['btc_price_avg'].mean():.2f}")
        print(f"  Min BTC price: ${final_df['btc_price_min'].min():.2f}")
        print(f"  Max BTC price: ${final_df['btc_price_max'].max():.2f}")

        # Show target variable statistics
        if 'btc_price_up' in final_df.columns:
            up_count = final_df['btc_price_up'].sum()
            total_with_target = final_df['btc_price_up'].notna().sum()
            print(f"\nTarget variable (btc_price_up) statistics:")
            print(f"  Articles with target: {total_with_target}")
            print(f"  Price went up: {up_count} ({up_count/total_with_target*100:.1f}%)")
            print(f"  Price went down: {total_with_target - up_count} ({(total_with_target - up_count)/total_with_target*100:.1f}%)")

    # Sort by date
    final_df = final_df.sort_values('date', ascending=False)

    # Save final dataset
    output_file = "bitcoin_news_final_dataset.csv"
    final_df.to_csv(output_file, index=False, encoding='utf-8')

    print("\n" + "="*80)
    print("FINAL DATASET SUMMARY")
    print("="*80)
    print(f"Total articles: {len(final_df)}")
    print(f"Columns: {len(final_df.columns)}")
    print(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
    print(f"\nKey columns:")
    for col in final_df.columns:
        non_null = final_df[col].notna().sum()
        print(f"  - {col}: {non_null} non-null values")

    print(f"\nOutput saved to: {output_file}")
    print(f"File size: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")
    print("="*80 + "\n")

    # Show sample
    print("Sample of the final dataset:")
    sample_cols = ['date', 'title', 'btc_price_avg', 'btc_price_change_pct', 'btc_price_up']
    available_cols = [col for col in sample_cols if col in final_df.columns]
    print(final_df[available_cols].head(10))

    print("\n" + "="*80)
    print("DATASET READY FOR ML TRAINING!")
    print("="*80)
    print("\nYou can now use this dataset to train a model that predicts:")
    print("  - btc_price_up: Binary classification (price goes up or down)")
    print("  - btc_price_change_pct: Regression (percentage price change)")
    print("\nFeatures to use:")
    print("  - title: News headline")
    print("  - description/content: News article text")
    print("  - sentiment: Sentiment analysis (if available)")
    print("  - btc_price_avg: Current BTC price")
    print("="*80 + "\n")

if __name__ == "__main__":
    create_final_dataset()
