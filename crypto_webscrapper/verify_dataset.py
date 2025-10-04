"""Verify the Bitcoin news dataset is ready for ML training."""

import pandas as pd
from pathlib import Path

def verify_dataset():
    """Verify the final dataset and provide statistics."""
    print("\n" + "="*80)
    print("DATASET VERIFICATION FOR ML TRAINING")
    print("="*80 + "\n")

    # Load the dataset
    dataset_file = "bitcoin_news_with_prices.csv"
    if not Path(dataset_file).exists():
        print(f"ERROR: {dataset_file} not found!")
        return

    print(f"Loading {dataset_file}...")
    df = pd.read_csv(dataset_file, low_memory=False)

    print("\n" + "-"*80)
    print("BASIC STATISTICS")
    print("-"*80)
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # Date range
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    print(f"Time span: {(df['date'].max() - df['date'].min()).days} days")

    # Feature columns
    print("\n" + "-"*80)
    print("FEATURE COLUMNS (for ML training)")
    print("-"*80)

    text_features = ['title', 'description', 'content']
    sentiment_features = ['sentiment', 'sentiment_positive', 'sentiment_neutral', 'sentiment_negative']
    price_features = ['btc_open', 'btc_high', 'btc_low', 'btc_close', 'btc_volume', 'btc_price_avg', 'btc_volatility_pct']
    target_features = ['btc_price_up', 'btc_price_change', 'btc_price_change_pct']

    print("\nText Features:")
    for col in text_features:
        if col in df.columns:
            non_null = df[col].notna().sum()
            pct = (non_null / len(df)) * 100
            avg_len = df[col].dropna().astype(str).str.len().mean()
            print(f"  {col:20} {non_null:7,} rows ({pct:5.1f}%) - avg length: {avg_len:.0f} chars")

    print("\nSentiment Features:")
    for col in sentiment_features:
        if col in df.columns:
            non_null = df[col].notna().sum()
            pct = (non_null / len(df)) * 100
            print(f"  {col:25} {non_null:7,} rows ({pct:5.1f}%)")

    print("\nPrice Features:")
    for col in price_features:
        if col in df.columns:
            non_null = df[col].notna().sum()
            pct = (non_null / len(df)) * 100
            if non_null > 0:
                mean_val = df[col].mean()
                print(f"  {col:25} {non_null:7,} rows ({pct:5.1f}%) - mean: {mean_val:.2f}")
            else:
                print(f"  {col:25} {non_null:7,} rows ({pct:5.1f}%)")

    # Target variables
    print("\n" + "-"*80)
    print("TARGET VARIABLES (what we want to predict)")
    print("-"*80)

    for col in target_features:
        if col in df.columns:
            non_null = df[col].notna().sum()
            pct = (non_null / len(df)) * 100
            print(f"\n{col}:")
            print(f"  Non-null values: {non_null:,} ({pct:.1f}%)")

            if col == 'btc_price_up':
                # Binary classification target
                if non_null > 0:
                    value_counts = df[col].value_counts()
                    print(f"  Distribution:")
                    for val, count in value_counts.items():
                        label = "UP" if val == 1 else "DOWN"
                        print(f"    {label}: {count:,} ({count/non_null*100:.1f}%)")
            else:
                # Regression targets
                if non_null > 0:
                    print(f"  Mean: {df[col].mean():.4f}")
                    print(f"  Std: {df[col].std():.4f}")
                    print(f"  Min: {df[col].min():.4f}")
                    print(f"  Max: {df[col].max():.4f}")

    # Data quality check
    print("\n" + "-"*80)
    print("DATA QUALITY CHECK")
    print("-"*80)

    # Check for rows with both features and targets
    has_text = df['title'].notna() | df['content'].notna()
    has_target = df['btc_price_up'].notna()
    usable_rows = (has_text & has_target).sum()

    print(f"\nUsable rows (has text AND target): {usable_rows:,} ({usable_rows/len(df)*100:.1f}%)")

    # Sample verification
    print("\n" + "-"*80)
    print("SAMPLE DATA VERIFICATION")
    print("-"*80)

    # Show a few complete rows
    sample_df = df[has_text & has_target].head(3)
    for idx, row in sample_df.iterrows():
        print(f"\nSample {idx + 1}:")
        print(f"  Date: {row['date']}")
        print(f"  Title: {row['title'][:80]}...")
        print(f"  BTC Close: ${row['btc_close']:.2f}")
        print(f"  Price Change: {row['btc_price_change_pct']:.2f}%")
        print(f"  Target (price_up): {row['btc_price_up']} ({'UP' if row['btc_price_up'] == 1 else 'DOWN'})")

    # Recommendations
    print("\n" + "="*80)
    print("DATASET READINESS ASSESSMENT")
    print("="*80)

    checks = []

    # Check 1: Sufficient data
    if usable_rows >= 1000:
        checks.append("✓ Sufficient training data (>1000 rows with features + targets)")
    elif usable_rows >= 500:
        checks.append("⚠ Moderate training data (500-1000 rows). Consider getting more data.")
    else:
        checks.append("✗ Insufficient training data (<500 rows). Need more data!")

    # Check 2: Text features
    text_coverage = (df['title'].notna() | df['content'].notna()).sum() / len(df) * 100
    if text_coverage >= 80:
        checks.append(f"✓ Good text feature coverage ({text_coverage:.1f}%)")
    else:
        checks.append(f"⚠ Low text feature coverage ({text_coverage:.1f}%)")

    # Check 3: Target balance
    if 'btc_price_up' in df.columns and df['btc_price_up'].notna().sum() > 0:
        up_pct = (df['btc_price_up'].sum() / df['btc_price_up'].notna().sum()) * 100
        if 40 <= up_pct <= 60:
            checks.append(f"✓ Balanced target classes (UP: {up_pct:.1f}%, DOWN: {100-up_pct:.1f}%)")
        else:
            checks.append(f"⚠ Imbalanced target classes (UP: {up_pct:.1f}%, DOWN: {100-up_pct:.1f}%)")

    # Check 4: Time range
    time_span = (df['date'].max() - df['date'].min()).days
    if time_span >= 365:
        checks.append(f"✓ Good time range ({time_span} days / {time_span/365:.1f} years)")
    else:
        checks.append(f"⚠ Limited time range ({time_span} days)")

    print("\nReadiness Checks:")
    for check in checks:
        print(f"  {check}")

    # Final recommendation
    print("\n" + "-"*80)
    if all('✓' in check for check in checks):
        print("✓ DATASET IS READY FOR ML TRAINING!")
    elif any('✗' in check for check in checks):
        print("✗ DATASET NEEDS IMPROVEMENT BEFORE TRAINING")
    else:
        print("⚠ DATASET IS USABLE BUT COULD BE IMPROVED")

    print("\n" + "="*80)
    print("NEXT STEPS FOR ML MODEL DEVELOPMENT")
    print("="*80)
    print("""
1. Data Preprocessing:
   - Tokenize and vectorize text (title + content) using TF-IDF or embeddings
   - Handle missing values in sentiment features
   - Normalize price features

2. Feature Engineering:
   - Extract keywords from news titles
   - Use sentiment scores as features
   - Add temporal features (day of week, month)
   - Calculate news volume per day

3. Model Selection:
   - Start with LogisticRegression or RandomForest for baseline
   - Try LSTM/GRU for sequential news analysis
   - Consider BERT/FinBERT for text embeddings

4. Training Strategy:
   - Time-based train/test split (e.g., 80% train, 20% test)
   - Use cross-validation with time series splits
   - Evaluate with accuracy, F1-score, ROC-AUC

5. Target Variables:
   - Binary Classification: btc_price_up (0 or 1)
   - Regression: btc_price_change_pct (percentage change)
    """)
    print("="*80 + "\n")

if __name__ == "__main__":
    verify_dataset()
