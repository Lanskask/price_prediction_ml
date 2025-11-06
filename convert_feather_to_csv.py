
import pandas as pd

def convert_feather_to_csv(feather_path, csv_path):
    """
    Converts a feather file to a csv file.
    """
    df = pd.read_feather(feather_path)
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    feather_file = "/Users/a1/Programming/ML/price_prediction_ml/crypto_webscrapper/bitcoin_news_final_dataset_hourly.feather"
    csv_file = "/Users/a1/Programming/ML/price_prediction_ml/crypto_webscrapper/bitcoin_news_final_dataset_hourly.csv"
    convert_feather_to_csv(feather_file, csv_file)
    print(f"Successfully converted {feather_file} to {csv_file}")
