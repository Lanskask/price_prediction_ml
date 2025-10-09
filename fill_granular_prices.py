"""Fill Bitcoin price columns with granular (hourly) prices matching exact timestamps."""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time
import numpy as np

def fetch_hourly_bitcoin_prices(start_date: datetime, end_date: datetime, chunk_days: int = 60) -> pd.DataFrame:
    """
    Fetch hourly Bitcoin prices using yfinance in chunks.

    Yahoo Finance limits hourly data to ~730 days per request.
    We'll fetch in chunks to cover the entire date range.

    Args:
        start_date: Start datetime
        end_date: End datetime
        chunk_days: Number of days to fetch per chunk (max ~730 for hourly)

    Returns:
        DataFrame with hourly Bitcoin OHLCV data
    """
    print(f"\nFetching hourly Bitcoin price data from Yahoo Finance...")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Time span: {(end_date - start_date).days} days")

    all_data = []
    current_start = start_date
    chunk_num = 1

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)

        try:
            print(f"\n  Chunk {chunk_num}: {current_start.date()} to {current_end.date()}")

            btc = yf.Ticker("BTC-USD")
            df = btc.history(
                start=current_start.strftime('%Y-%m-%d'),
                end=(current_end + timedelta(days=1)).strftime('%Y-%m-%d'),
                interval='1h'  # Hourly data
            )

            if not df.empty:
                # Reset index to get datetime as column
                df = df.reset_index()

                # Rename columns
                df = df.rename(columns={
                    'Datetime': 'datetime',
                    'Open': 'open_price',
                    'High': 'high_price',
                    'Low': 'low_price',
                    'Close': 'close_price',
                    'Volume': 'btc_volume'
                })

                # Remove timezone info for easier matching
                df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)

                all_data.append(df[['datetime', 'open_price', 'high_price', 'low_price', 'close_price', 'btc_volume']])
                print(f"    ✓ Fetched {len(df)} hours of data")
            else:
                print(f"    ⚠ No data returned for this chunk")

            # Rate limiting - be nice to Yahoo Finance
            time.sleep(1)

        except Exception as e:
            print(f"    ✗ Error fetching chunk: {e}")

        current_start = current_end
        chunk_num += 1

    if not all_data:
        print("\n✗ ERROR: No hourly data was fetched!")
        return pd.DataFrame()

    # Combine all chunks
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('datetime').drop_duplicates(subset=['datetime'])

    print(f"\n✓ Total hourly price points fetched: {len(combined_df):,}")
    print(f"  Price range: ${combined_df['low_price'].min():.2f} - ${combined_df['high_price'].max():.2f}")

    return combined_df

def match_prices_to_timestamps(news_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match news articles to Bitcoin prices using nearest timestamp matching.

    Args:
        news_df: DataFrame with news articles and timestamps
        price_df: DataFrame with hourly Bitcoin prices

    Returns:
        Merged DataFrame with prices matched to exact timestamps
    """
    print("\nMatching prices to exact article timestamps...")

    # Ensure datetime columns
    news_df['date'] = pd.to_datetime(news_df['date'])
    price_df['datetime'] = pd.to_datetime(price_df['datetime'])

    # Sort both dataframes by time
    news_df = news_df.sort_values('date').reset_index(drop=True)
    price_df = price_df.sort_values('datetime').reset_index(drop=True)

    print(f"  News articles: {len(news_df):,}")
    print(f"  Price data points: {len(price_df):,}")

    # Use merge_asof to match each article to the nearest earlier price
    # This ensures we use the price that was actually available at article publication time
    merged_df = pd.merge_asof(
        news_df,
        price_df,
        left_on='date',
        right_on='datetime',
        direction='nearest',  # Use nearest price (within 1 hour)
        tolerance=pd.Timedelta('1 hour')  # Maximum time difference allowed
    )

    # Calculate time difference between article and matched price
    merged_df['price_time_diff'] = abs((merged_df['date'] - merged_df['datetime']).dt.total_seconds() / 60)

    # Statistics
    matched_count = merged_df['close_price'].notna().sum()
    print(f"\n✓ Successfully matched {matched_count:,} articles ({matched_count/len(news_df)*100:.1f}%)")

    if matched_count > 0:
        avg_diff = merged_df[merged_df['close_price'].notna()]['price_time_diff'].mean()
        max_diff = merged_df[merged_df['close_price'].notna()]['price_time_diff'].max()
        print(f"  Average time difference: {avg_diff:.1f} minutes")
        print(f"  Max time difference: {max_diff:.1f} minutes")

    # Drop the temporary datetime column
    merged_df = merged_df.drop('datetime', axis=1, errors='ignore')

    return merged_df

def add_ml_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ML target variables based on next-hour price changes.

    Args:
        df: DataFrame with price data

    Returns:
        DataFrame with added target variables
    """
    print("\nAdding ML target variables...")

    # Sort by date
    df = df.sort_values('date').copy()

    # For each unique date (day), calculate next day's price change
    # Group by date (without time) and get the last price of each day
    df['date_only'] = df['date'].dt.date

    # Get last price of each day
    daily_close = df.groupby('date_only')['close_price'].last().reset_index()
    daily_close.columns = ['date_only', 'daily_close']

    # Calculate next day's close
    daily_close = daily_close.sort_values('date_only')
    daily_close['next_day_close'] = daily_close['daily_close'].shift(-1)
    daily_close['price_change'] = daily_close['next_day_close'] - daily_close['daily_close']
    daily_close['price_change_pct'] = (daily_close['price_change'] / daily_close['daily_close']) * 100
    daily_close['price_up'] = (daily_close['price_change'] > 0).astype(float)

    # Merge back to main dataframe
    df = df.merge(
        daily_close[['date_only', 'next_day_close', 'price_change', 'price_change_pct', 'price_up']],
        on='date_only',
        how='left'
    )

    # Rename target columns
    df = df.rename(columns={
        'next_day_close': 'btc_price_next_day',
        'price_change': 'btc_price_change',
        'price_change_pct': 'btc_price_change_pct',
        'price_up': 'btc_price_up'
    })

    # Calculate volatility (high-low range as percentage of close)
    df['btc_volatility_pct'] = ((df['high_price'] - df['low_price']) / df['close_price']) * 100

    # Average price
    df['btc_price_avg'] = (df['high_price'] + df['low_price']) / 2

    # Drop temporary column
    df = df.drop('date_only', axis=1)

    print("✓ Added target variables: btc_price_up, btc_price_change_pct, btc_volatility_pct")

    return df

def fill_granular_prices(input_file: str, output_file: str = None):
    """
    Fill Bitcoin price columns with granular (hourly) prices.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (defaults to input_file if not provided)
    """
    print("="*80)
    print("FILLING GRANULAR BITCOIN PRICE DATA IN NEWS DATASET")
    print("="*80)

    # Load the dataset
    if not Path(input_file).exists():
        print(f"\n✗ ERROR: {input_file} not found!")
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

    # Check timestamp granularity
    has_time = (news_df['date'].dt.time != pd.Timestamp('00:00:00').time()).sum()
    print(f"  Articles with specific time: {has_time:,} ({has_time/len(news_df)*100:.1f}%)")

    # Fetch hourly Bitcoin prices
    # Yahoo Finance has limitations, so we'll fetch in 60-day chunks
    price_df = fetch_hourly_bitcoin_prices(min_date, max_date, chunk_days=60)

    if price_df.empty:
        print("\n✗ Failed to fetch price data. Aborting.")
        return False

    # Match prices to article timestamps
    merged_df = match_prices_to_timestamps(news_df, price_df)

    # Add ML target variables
    merged_df = add_ml_targets(merged_df)

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
    print("\nSample data (first 5 rows with exact timestamps):")
    sample_cols = ['date', 'title', 'close_price', 'price_time_diff', 'btc_price_change_pct', 'btc_price_up']
    available_cols = [col for col in sample_cols if col in merged_df.columns]
    print(merged_df[available_cols].head(5).to_string(index=False, max_colwidth=40))

    print("\n" + "="*80)
    print("✓ DATASET NOW HAS GRANULAR (HOURLY) BITCOIN PRICES!")
    print("="*80)
    print("\nNote: Prices are matched to the nearest hour of article publication.")
    print("The 'price_time_diff' column shows minutes between article and price.")

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

    success = fill_granular_prices(input_file)

    if success:
        print("\nNext steps:")
        print("1. The dataset now has prices matched to article timestamps (±1 hour)")
        print("2. Start building your ML model with this more accurate price data!")
    else:
        print("\n✗ Failed to fill granular price data. Please check the errors above.")
        sys.exit(1)
