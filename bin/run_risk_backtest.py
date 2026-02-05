import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

from chronos import Chronos2Pipeline
from tiingo_data.download_data import get_daily_returns_data_cached
from utils import get_device
from core.risk_backtest import run_risk_backtest


def main():
    device = get_device()
    df = get_daily_returns_data_cached().dropna()

    df = df.iloc[-1200:]

    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=str(device),
        torch_dtype=torch.float32,
    )

    results = run_risk_backtest(
        pipeline=pipeline,
        df_returns=df,
        context_length=200,
        var_alphas=(0.1,),
        interval_pairs=((0.1, 0.9),),
        conformal_target_interval=(0.1, 0.9),
        conformal_target_coverage=0.8,
        calib_days=250,
    )

    print("RAW:", results["summary_raw"])
    if "summary_conformal" in results:
        print("CONFORMAL:", results["summary_conformal"])
        print("delta:", results["conformal_delta"])
        

if __name__ == "__main__":
    main()

