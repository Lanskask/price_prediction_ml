# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bitcoin price prediction ML project that combines cryptocurrency news from multiple sources with historical price data to train models predicting price movements. The project creates ML-ready datasets with labeled targets for both binary classification (price up/down) and regression (price change percentage).

## Development Commands

### Environment Setup
```bash
# Install dependencies using Poetry
poetry install

# Optional: Install with Kaggle support
poetry install --extras kaggle

# Use existing Python environment
poetry env use ~/Programming/Python/.venv/bin/python
poetry install
```

### Data Pipeline (Run in sequence)

1. **Download historical news from Kaggle:**
   ```bash
   poetry run download-news
   ```
   - Combines Bitcoin news from 4 Kaggle datasets
   - Filters and deduplicates articles
   - Output: `bitcoin_news_combined.csv`

2. **Scrape recent news:**
   ```bash
   poetry run scrape-recent
   ```
   - Fetches latest Bitcoin news from CryptoCompare API and CoinDesk RSS
   - Output: `bitcoin_news_recent_scraped.csv`

3. **Create combined dataset:**
   ```bash
   poetry run create-dataset
   ```
   - Merges all news sources
   - Output: `bitcoin_news_final_dataset.csv`

4. **Add Bitcoin price data:**
   ```bash
   poetry run add-prices
   ```
   - Downloads BTC-USD OHLCV data from Yahoo Finance using yfinance
   - Calculates target variables (btc_price_up, btc_price_change_pct)
   - Output: `bitcoin_news_with_prices.csv` (final ML-ready dataset)

5. **Verify dataset quality:**
   ```bash
   poetry run verify-dataset
   ```
   - Validates data completeness and readiness for training
   - Shows statistics and sample data

### Alternative: Direct Python Execution
```bash
python crypto_webscrapper/download_bitcoin_news.py
python crypto_webscrapper/scrape_recent_news.py
python crypto_webscrapper/create_final_dataset.py
python crypto_webscrapper/add_price_data.py
python crypto_webscrapper/verify_dataset.py
```

## Architecture

### Data Pipeline Flow
```
1. Historical News Collection (Kaggle datasets)
   ├─ Bitcoin-News Dataset
   ├─ Crypto News + (filtered for BTC)
   ├─ Crypto News with Prices
   └─ Historical Crypto News 2013-2018

2. Recent News Scraping
   ├─ CryptoCompare API (free, no key)
   ├─ CoinDesk RSS feed
   └─ NewsData.io (optional, requires API key)

3. Dataset Combination & Deduplication
   └─ Merges all sources, removes duplicates by title

4. Price Data Integration
   ├─ Yahoo Finance (yfinance) for BTC-USD
   └─ Calculates target variables for ML

5. Final Dataset
   └─ News articles + OHLCV + ML targets
```

### Key Modules

**`download_bitcoin_news.py`** - Kaggle dataset downloader
- Downloads 4 Bitcoin news datasets from Kaggle
- Requires `kagglehub` (optional dependency)
- Standardizes columns across different dataset schemas
- Filters Bitcoin-specific articles from multi-coin datasets

**`scrape_recent_news.py`** - Recent news scraper
- CryptoCompare API: Paginated news fetching with rate limiting
- CoinDesk RSS: feedparser-based extraction
- NewsData.io: Optional API integration (requires key)
- All sources filter for Bitcoin-related content

**`create_final_dataset.py`** - Dataset combiner
- Merges historical (Kaggle) + recent (scraped) news
- Deduplicates by title
- Sorts by date descending

**`add_price_data.py`** - Price data merger (PRIMARY SCRIPT)
- Uses yfinance to fetch BTC-USD historical data
- Merges news with prices by date
- **Generates ML target variables:**
  - `btc_price_up`: Binary (1 if next-day price increase, 0 otherwise)
  - `btc_price_change_pct`: Percentage price change for regression
  - `btc_volatility_pct`: Daily volatility measure
- Creates `bitcoin_news_with_prices.csv` - the final training dataset

**`verify_dataset.py`** - Quality validator
- Checks data completeness and balance
- Validates text features, price data, and targets
- Provides ML readiness assessment and recommendations

### Target Variables for ML

The final dataset includes these prediction targets:
- **btc_price_up**: Binary classification (1 = price went up next day, 0 = down)
- **btc_price_change_pct**: Regression target (percentage price change)
- **btc_volatility_pct**: Volatility measure (high-low range as % of close)

Target calculation: Current day's news → Predict next day's price movement

### Dataset Schema

**Text Features:**
- `title`: News headline (primary feature)
- `description`: Article summary
- `content`: Full article text
- `source`: News source identifier

**Price Features:**
- `btc_open/high/low/close`: OHLCV data
- `btc_volume`: Trading volume
- `btc_price_avg`: Average of high/low

**Target Variables:**
- `btc_price_up`: Binary (UP/DOWN)
- `btc_price_change`: Absolute price change
- `btc_price_change_pct`: Percentage change

**Metadata:**
- `date`: Publication date
- `url`: Article URL
- Sentiment columns (if available from source)

## Important Notes

### Data Dependencies
- Kaggle datasets require authentication (kagglehub handles this)
- News APIs are free but CryptoCompare/CoinDesk have rate limits
- Yahoo Finance (via yfinance) is free and reliable for BTC price data

### Data Quality Expectations
According to README notes, typical final dataset has:
- ~9,000 articles spanning 3-4 years
- ~99.5% usable rows with both text and price targets
- Balanced classes for binary classification (~48-52% split)

### Development Workflow
Always run the data pipeline in sequence (download → scrape → create → add-prices → verify) as each step depends on the previous one's output files.

### API Keys (Optional)
NewsData.io API key can be passed to `scrape_recent_news.py` for additional news sources, but the pipeline works without it using CryptoCompare and CoinDesk.
