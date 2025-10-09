"""Fill Bitcoin price columns with granular prices using Binance API."""

import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time
import numpy as np

def fetch_binance_hourly_prices(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch hourly Bitcoin prices from Binance API.

    Binance provides historical kline/candlestick data for free.
    Each request can fetch up to 1000 candles.

    Args:
        start_date: Start datetime
        end_date: End datetime

    Returns:
        DataFrame with hourly Bitcoin OHLCV data
    """
    print(f"\nFetching hourly Bitcoin price data from Binance API...")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Time span: {(end_date - start_date).days} days")

    all_data = []
    current_start = start_date

    # Binance returns 1000 candles per request, hourly = ~41 days
    chunk_hours = 1000
    chunk_num = 1

    # Binance klines endpoint
    url = "https://api.binance.com/api/v3/klines"

    while current_start < end_date:
        current_end = min(current_start + timedelta(hours=chunk_hours), end_date)

        try:
            print(f"\n  Chunk {chunk_num}: {current_start} to {current_end}")

            # Convert to milliseconds timestamp
            start_ms = int(current_start.timestamp() * 1000)
            end_ms = int(current_end.timestamp() * 1000)

            params = {
                'symbol': 'BTCUSDT',
                'interval': '1h',  # 1 hour candles
                'startTime': start_ms,
                'endTime': end_ms,
                'limit': 1000  # Max candles per request
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data and len(data) > 0:
                # Binance returns: [Open time, Open, High, Low, Close, Volume, Close time, ...]
                df = pd.DataFrame(data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])

                # Convert to appropriate types
                df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
                df['open_price'] = df['open'].astype(float)
                df['high_price'] = df['high'].astype(float)
                df['low_price'] = df['low'].astype(float)
                df['close_price'] = df['close'].astype(float)
                df['btc_volume'] = df['volume'].astype(float)

                # Keep only relevant columns
                df = df[['datetime', 'open_price', 'high_price', 'low_price', 'close_price', 'btc_volume']]

                all_data.append(df)
                print(f"    ✓ Fetched {len(df)} hours of data")

                # Move to next chunk (use last timestamp from this chunk)
                if len(df) > 0:
                    last_time = df['datetime'].max()
                    current_start = last_time + timedelta(hours=1)
                else:
                    current_start = current_end

            else:
                print(f"    ⚠ No data returned for this chunk")
                current_start = current_end

            # Rate limiting - Binance allows ~1200 requests per minute
            time.sleep(0.2)  # ~5 requests per second to be safe

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"    ⚠ Rate limited. Waiting 60 seconds...")
                time.sleep(60)
                continue
            else:
                print(f"    ✗ HTTP Error: {e}")
                current_start = current_end
        except Exception as e:
            print(f"    ✗ Error fetching chunk: {e}")
            current_start = current_end

        chunk_num += 1

    if not all_data:
        print("\n✗ ERROR: No price data was fetched!")
        return pd.DataFrame()

    # Combine all chunks
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('datetime').drop_duplicates(subset=['datetime'])

    print(f"\n✓ Total hourly price points fetched: {len(combined_df):,}")
    print(f"  Price range: ${combined_df['low_price'].min():.2f} - ${combined_df['high_price'].max():.2f}")
    print(f"  Date range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")

    return combined_df

def match_prices_to_timestamps(news_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match news articles to Bitcoin prices using nearest timestamp matching.

    Args:
        news_df: DataFrame with news articles and timestamps
        price_df: DataFrame with Bitcoin prices

    Returns:
        Merged DataFrame with prices matched to exact timestamps
    """
    print("\nMatching prices to exact article timestamps...")

    # Drop existing price columns if they exist to avoid conflicts
    price_cols_to_drop = ['open_price', 'high_price', 'low_price', 'close_price', 'btc_volume',
                          'btc_price_avg', 'btc_price_next_day', 'btc_price_change',
                          'btc_price_change_pct', 'btc_price_up', 'btc_volatility_pct']
    for col in price_cols_to_drop:
        if col in news_df.columns:
            news_df = news_df.drop(col, axis=1)
            print(f"  Dropped existing column: {col}")

    # Ensure datetime columns
    news_df['date'] = pd.to_datetime(news_df['date'])
    price_df['datetime'] = pd.to_datetime(price_df['datetime'])

    # Sort both dataframes by time
    news_df = news_df.sort_values('date').reset_index(drop=True)
    price_df = price_df.sort_values('datetime').reset_index(drop=True)

    print(f"  News articles: {len(news_df):,}")
    print(f"  Price data points: {len(price_df):,}")

    # Use merge_asof to match each article to the nearest price
    merged_df = pd.merge_asof(
        news_df,
        price_df,
        left_on='date',
        right_on='datetime',
        direction='nearest',
        tolerance=pd.Timedelta('2 hours')  # Maximum time difference allowed
    )

    # Calculate time difference between article and matched price
    merged_df['price_time_diff_minutes'] = abs((merged_df['date'] - merged_df['datetime']).dt.total_seconds() / 60)

    # Statistics
    matched_count = merged_df['close_price'].notna().sum()
    print(f"\n✓ Successfully matched {matched_count:,} articles ({matched_count/len(news_df)*100:.1f}%)")

    if matched_count > 0:
        avg_diff = merged_df[merged_df['close_price'].notna()]['price_time_diff_minutes'].mean()
        max_diff = merged_df[merged_df['close_price'].notna()]['price_time_diff_minutes'].max()
        median_diff = merged_df[merged_df['close_price'].notna()]['price_time_diff_minutes'].median()
        print(f"  Average time difference: {avg_diff:.1f} minutes")
        print(f"  Median time difference: {median_diff:.1f} minutes")
        print(f"  Max time difference: {max_diff:.1f} minutes")

    # Drop the temporary datetime column
    merged_df = merged_df.drop('datetime', axis=1, errors='ignore')

    return merged_df

def add_ml_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ML target variables based on next-day price changes.

    Args:
        df: DataFrame with price data

    Returns:
        DataFrame with added target variables
    """
    print("\nAdding ML target variables...")

    # Sort by date
    df = df.sort_values('date').copy()

    # For each unique date (day), calculate next day's price change
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

    # Calculate volatility
    df['btc_volatility_pct'] = ((df['high_price'] - df['low_price']) / df['close_price']) * 100

    # Average price
    df['btc_price_avg'] = (df['high_price'] + df['low_price']) / 2

    # Drop temporary column
    df = df.drop('date_only', axis=1)

    print("✓ Added target variables: btc_price_up, btc_price_change_pct, btc_volatility_pct")

    return df

def fill_granular_prices(input_file: str, output_file: str = None):
    """
    Fill Bitcoin price columns with granular prices using Binance.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (defaults to input_file if not provided)
    """
    print("="*80)
    print("FILLING GRANULAR BITCOIN PRICE DATA (Binance API)")
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

    # Estimate time needed
    total_hours = int((max_date - min_date).total_seconds() / 3600)
    num_requests = total_hours // 1000 + 1
    estimated_seconds = num_requests * 0.2
    print(f"\nEstimated fetch time: ~{estimated_seconds:.0f} seconds ({num_requests} API requests)")

    # Fetch Bitcoin prices from Binance
    price_df = fetch_binance_hourly_prices(min_date, max_date)

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
    print("\nSample data (5 random rows with timestamps):")
    sample_cols = ['date', 'title', 'close_price', 'price_time_diff_minutes', 'btc_price_change_pct', 'btc_price_up']
    available_cols = [col for col in sample_cols if col in merged_df.columns]
    sample = merged_df[merged_df['close_price'].notna()].sample(min(5, len(merged_df)))
    print(sample[available_cols].to_string(index=False, max_colwidth=40))

    print("\n" + "="*80)
    print("✓ DATASET NOW HAS GRANULAR (HOURLY) BITCOIN PRICES!")
    print("="*80)
    print("\nNote: Prices are matched to the nearest hour of article publication.")
    print("The 'price_time_diff_minutes' column shows minutes between article and matched price.")

    return True

if __name__ == "__main__":
    input_file = "crypto_webscrapper/bitcoin_news_final_dataset.csv"

    success = fill_granular_prices(input_file)

    if success:
        print("\n✓ Success! Dataset now has hourly prices matched to article timestamps.")
        print("\nNext steps:")
        print("1. The dataset now has granular OHLCV prices from Binance")
        print("2. Each article is matched to the Bitcoin price at its publication time (±2 hours)")
        print("3. Start building your ML model with this time-accurate price data!")
    else:
        print("\n✗ Failed to fill granular price data. Please check the errors above.")
        sys.exit(1)
