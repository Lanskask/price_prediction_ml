"""Add Bitcoin price data to the news dataset using yfinance."""

import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

def fetch_bitcoin_prices_yfinance(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical Bitcoin prices using yfinance (free, no API key).

    Args:
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'

    Returns:
        DataFrame with Bitcoin prices
    """
    print(f"Fetching Bitcoin price data from Yahoo Finance...")
    print(f"Date range: {start_date} to {end_date}")

    try:
        # Download Bitcoin data (BTC-USD)
        btc = yf.Ticker("BTC-USD")
        df = btc.history(start=start_date, end=end_date)

        if df.empty:
            print("No price data returned from Yahoo Finance")
            return pd.DataFrame()

        # Reset index to get date as column
        df = df.reset_index()

        # Rename columns
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'btc_open',
            'High': 'btc_high',
            'Low': 'btc_low',
            'Close': 'btc_close',
            'Volume': 'btc_volume'
        })

        # Add average price
        df['btc_price_avg'] = (df['btc_high'] + df['btc_low']) / 2

        # Keep only relevant columns
        df = df[['date', 'btc_open', 'btc_high', 'btc_low', 'btc_close', 'btc_volume', 'btc_price_avg']]

        print(f"Fetched {len(df)} days of price data")
        print(f"Price range: ${df['btc_low'].min():.2f} - ${df['btc_high'].max():.2f}")

        return df

    except Exception as e:
        print(f"Error fetching Bitcoin prices: {e}")
        return pd.DataFrame()

def add_price_targets(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add target variables for ML prediction.

    Args:
        prices_df: DataFrame with Bitcoin prices

    Returns:
        DataFrame with added target variables
    """
    # Sort by date
    prices_df = prices_df.sort_values('date').copy()

    # Calculate next day price
    prices_df['btc_price_next_day'] = prices_df['btc_close'].shift(-1)

    # Calculate price change
    prices_df['btc_price_change'] = prices_df['btc_price_next_day'] - prices_df['btc_close']
    prices_df['btc_price_change_pct'] = (prices_df['btc_price_change'] / prices_df['btc_close']) * 100

    # Binary target: 1 if price goes up, 0 if down
    prices_df['btc_price_up'] = (prices_df['btc_price_change'] > 0).astype(int)

    # Volatility (high-low range as percentage of close)
    prices_df['btc_volatility_pct'] = ((prices_df['btc_high'] - prices_df['btc_low']) / prices_df['btc_close']) * 100

    return prices_df

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

    # Round to date only (remove time)
    news_df['date_only'] = news_df['date'].dt.date
    prices_df['date_only'] = prices_df['date'].dt.date

    # Merge
    merged_df = news_df.merge(
        prices_df.drop('date', axis=1),
        on='date_only',
        how='left'
    )

    # Drop temporary column
    merged_df = merged_df.drop('date_only', axis=1)

    return merged_df

def main():
    """Main function to add price data to news dataset."""
    print("\n" + "="*80)
    print("ADDING BITCOIN PRICE DATA TO NEWS DATASET")
    print("="*80 + "\n")

    # Load news data
    news_file = "bitcoin_news_final_dataset.csv"
    if not Path(news_file).exists():
        print(f"Error: {news_file} not found. Run create_final_dataset.py first.")
        return

    print(f"Loading {news_file}...")
    news_df = pd.read_csv(news_file, low_memory=False)
    print(f"Loaded {len(news_df)} articles")

    # Clean dates
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
    news_df = news_df.dropna(subset=['date'])

    # Get date range
    min_date = news_df['date'].min().strftime('%Y-%m-%d')
    max_date = news_df['date'].max().strftime('%Y-%m-%d')
    print(f"News date range: {min_date} to {max_date}\n")

    # Fetch prices
    prices_df = fetch_bitcoin_prices_yfinance(min_date, max_date)

    if prices_df.empty:
        print("\nFailed to fetch price data!")
        return

    # Add target variables
    print("\nAdding target variables for ML prediction...")
    prices_df = add_price_targets(prices_df)

    # Merge
    print("Merging news with prices...")
    final_df = merge_news_with_prices(news_df, prices_df)

    # Statistics
    print("\n" + "="*80)
    print("MERGE STATISTICS")
    print("="*80)
    print(f"Total articles: {len(final_df)}")
    print(f"Articles with price data: {final_df['btc_close'].notna().sum()}")
    print(f"Articles with target (btc_price_up): {final_df['btc_price_up'].notna().sum()}")

    if final_df['btc_close'].notna().sum() > 0:
        print(f"\nPrice statistics:")
        print(f"  Average BTC close: ${final_df['btc_close'].mean():.2f}")
        print(f"  Min BTC close: ${final_df['btc_close'].min():.2f}")
        print(f"  Max BTC close: ${final_df['btc_close'].max():.2f}")

    if final_df['btc_price_up'].notna().sum() > 0:
        up_count = final_df['btc_price_up'].sum()
        total = final_df['btc_price_up'].notna().sum()
        print(f"\nTarget distribution:")
        print(f"  Price went UP next day: {up_count} ({up_count/total*100:.1f}%)")
        print(f"  Price went DOWN next day: {total - up_count} ({(total - up_count)/total*100:.1f}%)")

    # Save
    output_file = "bitcoin_news_with_prices.csv"
    final_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\n" + "="*80)
    print("FINAL DATASET SAVED")
    print("="*80)
    print(f"File: {output_file}")
    print(f"Size: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Total rows: {len(final_df)}")
    print(f"Total columns: {len(final_df.columns)}")

    # Show sample
    print("\nSample data:")
    sample_cols = ['date', 'title', 'btc_close', 'btc_price_change_pct', 'btc_price_up']
    available_cols = [col for col in sample_cols if col in final_df.columns]
    print(final_df[available_cols].head(10).to_string())

    print("\n" + "="*80)
    print("READY FOR ML TRAINING!")
    print("="*80)
    print("\nRecommended workflow:")
    print("1. Load the dataset: pd.read_csv('bitcoin_news_with_prices.csv')")
    print("2. Use NLP on 'title' and 'content' columns for features")
    print("3. Use 'sentiment' columns if available")
    print("4. Target variable: 'btc_price_up' (binary classification)")
    print("5. Alternative target: 'btc_price_change_pct' (regression)")
    print("\nGood luck with your model training!")
    print("="*80 + "\n")

if __name__ == "__main__":
    # Install yfinance if needed
    try:
        import yfinance as yf
    except ImportError:
        print("Installing yfinance...")
        import subprocess
        subprocess.run(["uv", "pip", "install", "yfinance"], check=True)
        import yfinance as yf

    main()
