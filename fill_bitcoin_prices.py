"""Fill Bitcoin price columns in bitcoin_news_final_dataset.csv using historical data."""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import sys

def fetch_bitcoin_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical Bitcoin prices using yfinance.

    Args:
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'

    Returns:
        DataFrame with Bitcoin OHLCV data
    """
    print(f"\nFetching Bitcoin price data from Yahoo Finance...")
    print(f"Date range: {start_date} to {end_date}")

    try:
        # Download Bitcoin data (BTC-USD)
        btc = yf.Ticker("BTC-USD")

        # Add one day to end_date to ensure we get prices for the last day
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
        end_date_adj = end_dt.strftime('%Y-%m-%d')

        df = btc.history(start=start_date, end=end_date_adj)

        if df.empty:
            print("ERROR: No price data returned from Yahoo Finance")
            return pd.DataFrame()

        # Reset index to get date as column
        df = df.reset_index()

        # Rename columns to match dataset
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open_price',
            'High': 'high_price',
            'Low': 'low_price',
            'Close': 'close_price',
            'Volume': 'btc_volume'
        })

        # Convert date to datetime and remove timezone info
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

        # Keep only relevant columns
        df = df[['date', 'open_price', 'high_price', 'low_price', 'close_price', 'btc_volume']]

        print(f"✓ Fetched {len(df)} days of price data")
        print(f"  Price range: ${df['low_price'].min():.2f} - ${df['high_price'].max():.2f}")

        return df

    except Exception as e:
        print(f"ERROR fetching Bitcoin prices: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def add_ml_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ML target variables to the dataset.

    Args:
        df: DataFrame with price data

    Returns:
        DataFrame with added target variables
    """
    print("\nAdding ML target variables...")

    # Sort by date to ensure proper calculations
    df = df.sort_values('date').copy()

    # Calculate next day price
    df['btc_price_next_day'] = df['close_price'].shift(-1)

    # Calculate price change
    df['btc_price_change'] = df['btc_price_next_day'] - df['close_price']
    df['btc_price_change_pct'] = (df['btc_price_change'] / df['close_price']) * 100

    # Binary target: 1 if price goes up, 0 if down
    df['btc_price_up'] = (df['btc_price_change'] > 0).astype(float)

    # Volatility (high-low range as percentage of close)
    df['btc_volatility_pct'] = ((df['high_price'] - df['low_price']) / df['close_price']) * 100

    # Average price
    df['btc_price_avg'] = (df['high_price'] + df['low_price']) / 2

    print("✓ Added target variables: btc_price_up, btc_price_change_pct, btc_volatility_pct")

    return df

def fill_price_columns(input_file: str, output_file: str = None):
    """
    Fill Bitcoin price columns in the dataset.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (defaults to input_file if not provided)
    """
    print("="*80)
    print("FILLING BITCOIN PRICE DATA IN NEWS DATASET")
    print("="*80)

    # Load the dataset
    if not Path(input_file).exists():
        print(f"\nERROR: {input_file} not found!")
        return False

    print(f"\nLoading {input_file}...")
    news_df = pd.read_csv(input_file, low_memory=False)
    print(f"✓ Loaded {len(news_df):,} articles")

    # Parse dates
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
    initial_count = len(news_df)
    news_df = news_df.dropna(subset=['date'])

    if len(news_df) < initial_count:
        print(f"  Removed {initial_count - len(news_df)} rows with invalid dates")

    # Get date range
    min_date = news_df['date'].min()
    max_date = news_df['date'].max()
    print(f"\nNews date range: {min_date} to {max_date}")
    print(f"  Time span: {(max_date - min_date).days} days")

    # Fetch Bitcoin prices
    price_df = fetch_bitcoin_prices(
        min_date.strftime('%Y-%m-%d'),
        max_date.strftime('%Y-%m-%d')
    )

    if price_df.empty:
        print("\nFailed to fetch price data. Aborting.")
        return False

    # Add target variables to price data
    price_df = add_ml_targets(price_df)

    # Extract just the date (no time) for merging
    news_df['date_only'] = news_df['date'].dt.date
    price_df['date_only'] = price_df['date'].dt.date

    # Merge news with prices
    print("\nMerging news with Bitcoin prices by date...")

    # Select price columns to merge
    price_cols = ['date_only', 'open_price', 'high_price', 'low_price', 'close_price',
                  'btc_volume', 'btc_price_avg', 'btc_price_next_day', 'btc_price_change',
                  'btc_price_change_pct', 'btc_price_up', 'btc_volatility_pct']

    merged_df = news_df.merge(
        price_df[price_cols],
        on='date_only',
        how='left',
        suffixes=('', '_new')
    )

    # Update the original price columns
    for col in ['open_price', 'high_price', 'low_price', 'close_price']:
        if col + '_new' in merged_df.columns:
            merged_df[col] = merged_df[col + '_new']
            merged_df = merged_df.drop(col + '_new', axis=1)

    # Drop temporary date_only column
    merged_df = merged_df.drop('date_only', axis=1)

    # Sort by date descending (most recent first)
    merged_df = merged_df.sort_values('date', ascending=False)

    # Statistics
    print("\n" + "="*80)
    print("MERGE STATISTICS")
    print("="*80)
    print(f"Total articles: {len(merged_df):,}")
    print(f"Articles with price data: {merged_df['close_price'].notna().sum():,} ({merged_df['close_price'].notna().sum()/len(merged_df)*100:.1f}%)")
    print(f"Articles with target (btc_price_up): {merged_df['btc_price_up'].notna().sum():,} ({merged_df['btc_price_up'].notna().sum()/len(merged_df)*100:.1f}%)")

    if merged_df['close_price'].notna().sum() > 0:
        print(f"\nPrice statistics:")
        print(f"  Average BTC close: ${merged_df['close_price'].mean():.2f}")
        print(f"  Min BTC close: ${merged_df['close_price'].min():.2f}")
        print(f"  Max BTC close: ${merged_df['close_price'].max():.2f}")

    if merged_df['btc_price_up'].notna().sum() > 0:
        up_count = merged_df['btc_price_up'].sum()
        total = merged_df['btc_price_up'].notna().sum()
        print(f"\nTarget distribution (for ML classification):")
        print(f"  Price went UP next day: {int(up_count):,} ({up_count/total*100:.1f}%)")
        print(f"  Price went DOWN next day: {int(total - up_count):,} ({(total - up_count)/total*100:.1f}%)")

    # Save the updated dataset
    if output_file is None:
        output_file = input_file

    merged_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\n" + "="*80)
    print("DATASET SAVED SUCCESSFULLY")
    print("="*80)
    print(f"File: {output_file}")
    print(f"Size: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Total rows: {len(merged_df):,}")
    print(f"Total columns: {len(merged_df.columns)}")

    # Show sample
    print("\nSample data (first 3 rows):")
    sample_cols = ['date', 'title', 'close_price', 'btc_price_change_pct', 'btc_price_up']
    available_cols = [col for col in sample_cols if col in merged_df.columns]
    print(merged_df[available_cols].head(3).to_string(index=False))

    print("\n" + "="*80)
    print("✓ DATASET IS NOW READY FOR ML TRAINING!")
    print("="*80)

    return True

if __name__ == "__main__":
    input_file = "crypto_webscrapper/bitcoin_news_final_dataset.csv"

    # Install yfinance if needed
    try:
        import yfinance as yf
    except ImportError:
        print("Installing yfinance...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "yfinance"], check=True)
        import yfinance as yf

    success = fill_price_columns(input_file)

    if success:
        print("\nNext steps:")
        print("1. Use 'poetry run verify-dataset' to validate the data")
        print("2. Start building your ML model with the updated dataset!")
    else:
        print("\nFailed to fill price data. Please check the errors above.")
        sys.exit(1)
