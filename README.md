# Bitcoin Price Prediction ML

Machine learning project for predicting Bitcoin price movements using news sentiment analysis and historical price data.

## Overview

This project combines cryptocurrency news articles from multiple sources with Bitcoin price data to train ML models that predict price movements.

## Features

- **Multi-source news aggregation**: Fetches Bitcoin news from CryptoCompare, CoinDesk, Kaggle datasets, and NewsData.io
- **Historical price data**: Downloads Bitcoin OHLCV data using Yahoo Finance
- **Sentiment analysis**: Processes news sentiment to correlate with price movements
- **ML-ready dataset**: Generates labeled datasets for binary classification (price up/down) and regression tasks

## Project Structure

```
price_prediction_ml/
├── crypto_webscrapper/       # News scraping and dataset creation
└── pyproject.toml            # Project dependencies
```

## Installation

```bash
# Install dependencies
poetry install

# Optional: Install Kaggle dependencies
poetry install --extras kaggle
```

To use the existing Python environment at `~/Programming/Python/.venv`:
```bash
poetry env use ~/Programming/Python/.venv/bin/python
poetry install
```

## Usage

The project provides convenient CLI commands via Poetry:

### 1. Download historical Bitcoin news from Kaggle

```bash
poetry run download-news
```

### 2. Scrape recent Bitcoin news

```bash
poetry run scrape-recent
```

### 3. Create final combined dataset

```bash
poetry run create-dataset
```

### 4. Add Bitcoin price data

```bash
poetry run add-prices
```

### 5. Verify the dataset

```bash
poetry run verify-dataset
```

Alternatively, run Python scripts directly:
```bash
python crypto_webscrapper/download_bitcoin_news.py
python crypto_webscrapper/scrape_recent_news.py
python crypto_webscrapper/create_final_dataset.py
python crypto_webscrapper/add_price_data.py
python crypto_webscrapper/verify_dataset.py
```

## Dependencies

- **pandas**: Data manipulation and analysis
- **requests**: HTTP requests for API calls
- **yfinance**: Bitcoin price data from Yahoo Finance
- **feedparser**: RSS feed parsing
- **tqdm**: Progress bars
- **aiohttp**: Async HTTP requests
- **karpet**: Cryptocurrency news scraping
- **kagglehub** (optional): Kaggle dataset downloads

## Output

The final dataset (`bitcoin_news_with_prices.csv`) includes:

- News article metadata (date, title, description, content, source)
- Bitcoin OHLCV data (open, high, low, close, volume)
- Target variables:
  - `btc_price_up`: Binary (1 if price increased next day, 0 otherwise)
  - `btc_price_change_pct`: Percentage price change for regression
  - `btc_volatility_pct`: Daily volatility measure

## ML Training Recommendations

1. Use NLP techniques (TF-IDF, embeddings) on `title` and `content` columns
2. Incorporate sentiment scores if available
3. Target variable: `btc_price_up` for classification or `btc_price_change_pct` for regression
4. Consider time-series cross-validation to avoid data leakage

## License

MIT
