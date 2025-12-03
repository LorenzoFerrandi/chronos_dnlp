import pandas as pd
from tiingo import TiingoClient

TIINGO_API_KEY = "63a212473489b88a0406ce83cad2a801ef188bf0"
TIINGO_BASE_URL = "https://api.tiingo.com/tiingo/daily/"
TIINGO_CONFIG = {"session": True, "api_key": TIINGO_API_KEY}


def get_adj_close_px(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    tiingo = TiingoClient(TIINGO_CONFIG)
    return tiingo.get_dataframe(
        tickers, metric_name="adjClose", startDate=start_date, endDate=end_date
    )


def get_daily_returns(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    data = get_adj_close_px(tickers, start_date, end_date)
    data = data.pct_change()
    return data


if __name__ == "__main__":
    df_ret = get_daily_returns(["AAPL"], start_date="2005-01-01", end_date="2025-11-20")
    print(df_ret.head())
    print(df_ret.tail())


# tiingo.get_dataframe(["AAPL", "MSFT", "VST", "GOOG", "RACE", "JPM", "GS", "NVDA", "CRM", "LLY", "TSLA"], metric_name="adjClose", startDate="2012-01-01", endDate="2025-11-20").pct_change().cumsum().plot()
