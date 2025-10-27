"""
Iteration: Outer 2, Inner 2
Timestamp: 20251026_135341
Execution Status: SUCCESS
Feedback from Planner:
No feedback yet
"""

"""
Engineer Script – Iteration 2 (Rev-A)
Adds benchmark comparison (SPY, SOXX) to existing Nvidia (NVDA) analysis.
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import matplotlib
matplotlib.use('Agg')  # non-interactive backend

import os
import json
import logging
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TICKERS = ["NVDA", "SPY", "SOXX"]
START_DATE = "2025-09-23"
END_DATE   = "2025-10-23"
DATA_DIR = "data"
FIG_DIR  = "figures"

COLOR_MAP = {
    "NVDA": "#76B900",
    "SPY":  "#1f77b4",
    "SOXX": "#ff7f0e",
}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

def fetch_price_data(ticker: str) -> pd.DataFrame:
    """
    Fetch price data for one ticker, fallback to Close for Adj Close if absent.
    """
    logging.info(f"Fetching {ticker}")
    data = (
        yf.Ticker(ticker)
        .history(start=START_DATE, end=END_DATE, auto_adjust=False)
        .loc[:, ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    )

    if "Adj Close" not in data.columns or data["Adj Close"].isna().all():
        logging.warning(f"{ticker}: 'Adj Close' missing – copying from 'Close'.")
        data["Adj Close"] = data["Close"]

    csv_path = f"{DATA_DIR}/{ticker.lower()}_{START_DATE}_to_{END_DATE}.csv"
    data.to_csv(csv_path)
    logging.info(f"{ticker}: data saved → {csv_path}")
    return data

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def annualised_vol(daily_ret: pd.Series) -> float:
    return float(daily_ret.std(ddof=0) * np.sqrt(252) * 100)

def pct_return(start: float, end: float) -> float:
    return round((end / start - 1) * 100, 2)

# --------------------------------------------------------------------------- #
# Analyzer
# --------------------------------------------------------------------------- #
class StockAnalyzer:
    def __init__(self, df: pd.DataFrame, symbol: str):
        self.df = df.copy()
        self.symbol = symbol
        self.metrics: Dict = {}

    def basic_metrics(self) -> None:
        start_p = float(self.df["Adj Close"].iloc[0])
        end_p   = float(self.df["Adj Close"].iloc[-1])
        daily   = self.df["Adj Close"].pct_change().dropna()

        self.metrics.update({
            "Start_Adj_Close": round(start_p, 2),
            "End_Adj_Close":   round(end_p, 2),
            "Period_Return_%": pct_return(start_p, end_p),
            "Volatility_%":    round(annualised_vol(daily), 1),
        })

    # ---------------- NVDA only ---------------- #
    def nvda_extras(self) -> None:
        self.df["SMA_5"]  = self.df["Adj Close"].rolling(5).mean()
        self.df["SMA_20"] = self.df["Adj Close"].rolling(20).mean()
        self.df["RSI_14"] = compute_rsi(self.df["Adj Close"])

        hi, lo = self.df["High"].max(), self.df["Low"].min()
        avg_p  = self.df["Adj Close"].mean()
        avg_v  = int(self.df["Volume"].mean())

        daily_pct = self.df["Adj Close"].pct_change().dropna() * 100
        gains = (
            daily_pct.sort_values(ascending=False).head(3).round(2)
            .to_frame("pct_change").reset_index().rename(columns={"index": "date"})
        )
        losses = (
            daily_pct.sort_values().head(3).round(2)
            .to_frame("pct_change").reset_index().rename(columns={"index": "date"})
        )

        self.metrics.update({
            "High_Low_Avg": {
                "High": round(float(hi), 2),
                "Low":  round(float(lo), 2),
                "Average": round(float(avg_p), 2),
            },
            "Average_Volume": avg_v,
            "Latest_Indicators": {
                "SMA_5":  round(float(self.df["SMA_5"].iloc[-1]), 2),
                "SMA_20": round(float(self.df["SMA_20"].iloc[-1]), 2),
                "RSI_14": round(float(self.df["RSI_14"].iloc[-1]), 2),
            },
            "Top_Gains_%": gains.to_dict("records"),
            "Top_Losses_%": losses.to_dict("records"),
        })

    def nvda_figures(self) -> Dict[str, str]:
        fig_dict = {}
        # Price + SMA
        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df["Adj Close"], label="Adj Close", color="#333")
        plt.plot(self.df.index, self.df["SMA_5"],  label="SMA 5",  color="#1f77b4")
        plt.plot(self.df.index, self.df["SMA_20"], label="SMA 20", color="#ff7f0e")
        plt.title("NVIDIA Price & Moving Averages – 1 Month")
        plt.xlabel("Date"); plt.ylabel("Price (USD)")
        plt.grid(alpha=.3); plt.legend()
        path_p = f"{FIG_DIR}/nvda_price_moving_avg.png"
        plt.savefig(path_p, dpi=120, bbox_inches="tight"); plt.close()
        fig_dict["Price_SMA"] = path_p

        # Volume
        plt.figure(figsize=(12, 4))
        plt.bar(self.df.index, self.df["Volume"], color="#76B900")
        plt.title("NVIDIA Trading Volume – 1 Month")
        plt.xlabel("Date"); plt.ylabel("Volume")
        plt.grid(axis="y", alpha=.3)
        path_v = f"{FIG_DIR}/nvda_volume.png"
        plt.savefig(path_v, dpi=120, bbox_inches="tight"); plt.close()
        fig_dict["Volume"] = path_v
        return fig_dict

# --------------------------------------------------------------------------- #
def benchmark_chart(adj_df: pd.DataFrame) -> str:
    idx = adj_df / adj_df.iloc[0] * 100
    plt.figure(figsize=(12, 6))
    for t in idx.columns:
        plt.plot(idx.index, idx[t], label=t, color=COLOR_MAP.get(t), linewidth=2)
    plt.title("Nvidia vs Benchmarks – 1-Month Indexed Performance (Start = 100)")
    plt.xlabel("Date"); plt.ylabel("Indexed Price (Start = 100)")
    plt.grid(alpha=.3); plt.legend()
    path = f"{FIG_DIR}/nvda_vs_benchmarks.png"
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    return path

# --------------------------------------------------------------------------- #
def main() -> None:
    ensure_dirs()

    # Fetch data
    dfs = {t: fetch_price_data(t) for t in TICKERS}

    # Combined Adj Close for benchmark chart
    adj_df = pd.concat({t: df["Adj Close"] for t, df in dfs.items()}, axis=1)
    adj_df.columns = TICKERS  # ensure order

    metrics: Dict = {}
    fig_paths: Dict = {}      # ensure exists for later merge

    # Analyse each ticker
    for t in TICKERS:
        a = StockAnalyzer(dfs[t], t)
        a.basic_metrics()
        if t == "NVDA":
            a.nvda_extras()
            fig_paths.update(a.nvda_figures())
        metrics[t] = a.metrics

    # Benchmark figure
    fig_paths["Benchmarks"] = benchmark_chart(adj_df)

    # Attach figure paths
    metrics["Figure_Paths"] = fig_paths

    print(json.dumps(metrics, indent=2, default=str))
    logging.info("Analysis complete.")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()