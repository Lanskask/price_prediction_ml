  Statistics:
  - 9,007 articles spanning 3.9 years (Nov 2021 - Oct 2025)
  - 8,959 usable rows (99.5%) with both news text and price targets
  - 5.6 MB dataset ready for training

  Features:
  - ✅ Text: title (100%), content (100%)
  - ✅ Price data: BTC open/high/low/close/volume (99.5%)
  - ✅ Target variables:
    - btc_price_up - Binary (UP: 48.3%, DOWN: 51.7%) ← Well balanced!
    - btc_price_change_pct - Regression (price change percentage)

  Data Sources:
  1. Kaggle datasets (40,651 articles from 4 datasets)
  2. Recent scraped news from CoinDesk
  3. Historical Bitcoin prices from Yahoo Finance

  📁 Files Created:

  1. bitcoin_news_with_prices.csv - Main dataset for training
  2. download_bitcoin_news.py - Downloads Kaggle datasets
  3. scrape_recent_news.py - Scrapes latest news
  4. add_price_data.py - Adds Bitcoin price data
  5. verify_dataset.py - Validates dataset quality

  🎯 News Events That Affect Bitcoin:

  ✅ Regulatory: SEC approvals, ETF decisions, country regulations✅ Institutional: Major company adoptions/rejections✅ Security: Exchange hacks, wallet breaches✅ Macroeconomic:
  Fed decisions, inflation, interest rates✅ Technical: Mining difficulty, halvings, network upgrades✅ Market: Whale movements, major liquidations