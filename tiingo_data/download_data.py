import pathlib
import os
import pandas as pd
from tiingo_data.tiingo import get_daily_returns
from tiingo_data.tiingo_tickers import (
    TICKERS,
)


DATA_DIR = "data"


def download_and_cache_data(start_date, end_date):
    for ticker in TICKERS:
        file_path = os.path.join(DATA_DIR, f"{ticker}.parquet")

        try:
            print(f"Downloading {ticker}...", end=" ")

            # Download entire available history
            df = get_daily_returns([ticker], start_date=start_date, end_date=end_date)

            # Reset index to make 'date' and 'symbol' columns (easier for filtering later)
            df = df.reset_index()

            # Save to Parquet
            df.to_parquet(file_path)
            print("Done.")

        except Exception as e:
            print(f"Failed: {e}")


def get_daily_returns_data_cached(ticker: str = None) -> pd.DataFrame:
    current_path = pathlib.Path(__file__).resolve()
    parent_dir = current_path.parent
    grandparent_dir = parent_dir.parent  # chronos_dnlp

    if ticker:
        # Load a single stock
        df = pd.read_parquet(grandparent_dir / f"{DATA_DIR}/{ticker}.parquet")
    else:
        # OR Load ALL stocks into one single DataFrame (fast with Parquet)
        all_files = [
            grandparent_dir / os.path.join(DATA_DIR, f)
            for f in os.listdir(grandparent_dir / DATA_DIR)
            if f.endswith(".parquet")
        ]
        df = pd.concat(
            [
                pd.read_parquet(f).set_index("index").rename_axis("date", axis=0)
                for f in all_files
            ],
            axis=1,
        )

    return df


if __name__ == "__main__":
    download_and_cache_data(start_date="2005-01-01", end_date="2025-11-20")

    # df = get_daily_returns_data_cached()
    # print(df.head())
    # print(df.tail())
