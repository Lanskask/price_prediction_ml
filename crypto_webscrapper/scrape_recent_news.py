"""Scrape recent Bitcoin news from CryptoCompare and NewsAPI for 2024-2025."""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

def fetch_cryptocompare_news(limit: int = 2000) -> List[Dict]:
    """
    Fetch Bitcoin news from CryptoCompare API (free, no API key required).

    Args:
        limit: Maximum number of articles to fetch

    Returns:
        List of news articles
    """
    url = "https://min-api.cryptocompare.com/data/v2/news/"
    params = {
        'categories': 'BTC',
        'lang': 'EN',
        'sortOrder': 'latest'
    }

    all_articles = []

    print(f"Fetching Bitcoin news from CryptoCompare...")

    try:
        # CryptoCompare returns latest news, we'll paginate through
        for page in range(0, limit, 100):
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('Response') == 'Success':
                articles = data.get('Data', [])
                if not articles:
                    break

                for article in articles:
                    all_articles.append({
                        'date': datetime.fromtimestamp(article.get('published_on', 0)),
                        'title': article.get('title', ''),
                        'description': article.get('body', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', ''),
                        'content': article.get('body', ''),
                        'tags': ','.join(article.get('tags', '').split('|')),
                        'image_url': article.get('imageurl', '')
                    })

                print(f"Fetched {len(all_articles)} articles so far...")

                # Update params for next page
                if articles:
                    last_timestamp = articles[-1].get('published_on', 0)
                    params['lTs'] = last_timestamp

                time.sleep(1)  # Rate limiting
            else:
                print(f"Error from API: {data.get('Message', 'Unknown error')}")
                break

        print(f"Total CryptoCompare articles: {len(all_articles)}")
        return all_articles

    except Exception as e:
        print(f"Error fetching from CryptoCompare: {e}")
        return all_articles

def fetch_newsdata_io(api_key: str = None) -> List[Dict]:
    """
    Fetch Bitcoin news from NewsData.io (requires API key, but has free tier).
    Get your free API key from: https://newsdata.io/

    Args:
        api_key: NewsData.io API key

    Returns:
        List of news articles
    """
    if not api_key:
        print("Skipping NewsData.io - no API key provided")
        print("You can get a free API key from https://newsdata.io/")
        return []

    url = "https://newsdata.io/api/1/news"
    params = {
        'apikey': api_key,
        'q': 'bitcoin OR btc',
        'language': 'en',
        'category': 'business,technology'
    }

    all_articles = []

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get('status') == 'success':
            articles = data.get('results', [])

            for article in articles:
                all_articles.append({
                    'date': article.get('pubDate', ''),
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('link', ''),
                    'source': article.get('source_id', ''),
                    'content': article.get('content', ''),
                    'image_url': article.get('image_url', '')
                })

            print(f"Total NewsData.io articles: {len(all_articles)}")

        return all_articles

    except Exception as e:
        print(f"Error fetching from NewsData.io: {e}")
        return []

def fetch_coindesk_rss() -> List[Dict]:
    """
    Fetch Bitcoin news from CoinDesk RSS feed.

    Returns:
        List of news articles
    """
    import feedparser

    feeds = [
        'https://www.coindesk.com/arc/outboundfeeds/rss/',
        'https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml'
    ]

    all_articles = []

    for feed_url in feeds:
        try:
            print(f"Fetching from CoinDesk RSS feed...")
            feed = feedparser.parse(feed_url)

            for entry in feed.entries:
                # Only include if Bitcoin is mentioned
                if 'bitcoin' in entry.title.lower() or 'btc' in entry.title.lower() or \
                   'bitcoin' in entry.get('summary', '').lower() or 'btc' in entry.get('summary', '').lower():
                    all_articles.append({
                        'date': entry.get('published', ''),
                        'title': entry.get('title', ''),
                        'description': entry.get('summary', ''),
                        'url': entry.get('link', ''),
                        'source': 'CoinDesk',
                        'content': entry.get('summary', '')
                    })

            print(f"Total CoinDesk articles: {len(all_articles)}")
            break  # If successful, no need to try other feeds

        except Exception as e:
            print(f"Error fetching from CoinDesk RSS: {e}")
            continue

    return all_articles

def scrape_all_sources(newsdata_api_key: str = None) -> pd.DataFrame:
    """
    Scrape Bitcoin news from all available sources.

    Args:
        newsdata_api_key: Optional API key for NewsData.io

    Returns:
        DataFrame with all scraped articles
    """
    print("\n" + "="*80)
    print("SCRAPING RECENT BITCOIN NEWS FROM MULTIPLE SOURCES")
    print("="*80 + "\n")

    all_articles = []

    # Fetch from CryptoCompare (free, no API key)
    cryptocompare_articles = fetch_cryptocompare_news(limit=2000)
    all_articles.extend(cryptocompare_articles)

    # Fetch from CoinDesk RSS
    coindesk_articles = fetch_coindesk_rss()
    all_articles.extend(coindesk_articles)

    # Fetch from NewsData.io (if API key provided)
    newsdata_articles = fetch_newsdata_io(newsdata_api_key)
    all_articles.extend(newsdata_articles)

    if not all_articles:
        print("\nNo articles were scraped!")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_articles)

    # Clean and standardize dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date', ascending=False)

    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['title'], keep='first')
    print(f"\nRemoved {initial_count - len(df)} duplicate articles")

    # Save to CSV
    output_file = "bitcoin_news_recent_scraped.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')

    print("\n" + "="*80)
    print("SCRAPING SUMMARY")
    print("="*80)
    print(f"Total articles scraped: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Output saved to: {output_file}")
    print("="*80 + "\n")

    return df

if __name__ == "__main__":
    # Install feedparser if needed
    try:
        import feedparser
    except ImportError:
        print("Installing feedparser...")
        import subprocess
        subprocess.run(["uv", "pip", "install", "feedparser"], check=True)
        import feedparser

    # Scrape from all sources
    # If you have a NewsData.io API key, pass it here:
    # scraped_df = scrape_all_sources(newsdata_api_key="YOUR_API_KEY")
    scraped_df = scrape_all_sources()

    # Show sample
    if not scraped_df.empty:
        print("\nSample of scraped data:")
        print(scraped_df[['date', 'title', 'source']].head(10))
