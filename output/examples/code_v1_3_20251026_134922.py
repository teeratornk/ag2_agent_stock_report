"""
Iteration: Outer 1, Inner 3
Timestamp: 20251026_134922
Execution Status: SUCCESS
Feedback from Planner:
No feedback yet
"""

# nvda_monthly_analysis.py
# Engineer revision – add latest indicators & tidy gain/loss lists (2025-10-23)

import matplotlib
matplotlib.use("Agg")          # Non-interactive backend

import os
import logging
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# --------------------------------------------------------------------------
# 0. Logging configuration
# --------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# 1. Global constants
# --------------------------------------------------------------------------
TICKER      = "NVDA"
START_DATE  = "2025-09-23"
END_DATE_EX = "2025-10-24"                 # yfinance end date is exclusive
RAW_CSV     = "data/nvda_20250923_20251023.csv"
PRICE_FIG   = "figures/nvda_price_moving_avg.png"
VOLUME_FIG  = "figures/nvda_volume.png"

# --------------------------------------------------------------------------
# 2. Helper functions
# --------------------------------------------------------------------------
def ensure_directories() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    logger.debug("Verified data/ and figures/ directories.")


def fetch_stock_data(ticker: str, start_date: str, end_date_exclusive: str) -> pd.DataFrame:
    logger.info(f"Fetching {ticker} data {start_date} → "
                f"{datetime.fromisoformat(end_date_exclusive) - timedelta(days=1):%Y-%m-%d}")
    df = yf.Ticker(ticker).history(start=start_date, end=end_date_exclusive)
    if df.empty:
        raise ValueError("yfinance returned no data.")
    df.index = df.index.tz_localize(None)

    # Ensure 'Adj Close' column exists
    if "Adj Close" not in df.columns:
        if "Close" in df.columns:
            df["Adj Close"] = df["Close"]
            logger.warning("'Adj Close' column missing; duplicated 'Close' into 'Adj Close'.")
        else:
            raise KeyError("Neither 'Adj Close' nor 'Close' found in data.")
    return df


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = np.where(delta > 0,  delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)

    gain_series = pd.Series(gain, index=series.index)
    loss_series = pd.Series(loss, index=series.index)

    avg_gain = gain_series.rolling(period, min_periods=period).mean()
    avg_loss = loss_series.rolling(period, min_periods=period).mean()
    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA_5"]  = df["Adj Close"].rolling(5).mean()
    df["SMA_20"] = df["Adj Close"].rolling(20).mean()
    df["RSI_14"] = calculate_rsi(df["Adj Close"])
    return df


def df_top_moves(df: pd.DataFrame, n: int = 3) -> (List[Dict], List[Dict]):
    """Return two lists of dicts for top n gains and losses."""
    gains_df  = df.nlargest(n, "PctChange")[["PctChange"]].round(2)
    losses_df = df.nsmallest(n, "PctChange")[["PctChange"]].round(2)

    gains_list  = [{"date": d.strftime("%Y-%m-%d"), "pct_change": v["PctChange"]}
                   for d, v in gains_df.iterrows()]
    losses_list = [{"date": d.strftime("%Y-%m-%d"), "pct_change": v["PctChange"]}
                   for d, v in losses_df.iterrows()]
    return gains_list, losses_list


def compute_summary_metrics(df: pd.DataFrame) -> dict:
    pct_change_full = (df["Adj Close"].iloc[-1] - df["Adj Close"].iloc[0]) / df["Adj Close"].iloc[0] * 100
    df["PctChange"] = df["Adj Close"].pct_change() * 100

    gains_list, losses_list = df_top_moves(df)

    latest = df.iloc[-1]
    summary = {
        # Period-wide stats
        "percentage_change_full_period_%": round(pct_change_full, 2),
        "max_adj_close": round(df["Adj Close"].max(), 2),
        "min_adj_close": round(df["Adj Close"].min(), 2),
        "average_adj_close": round(df["Adj Close"].mean(), 2),
        "average_daily_volume": int(df["Volume"].mean()),
        # Latest indicators (as of last row)
        "latest_adj_close": round(latest["Adj Close"], 2),
        "latest_SMA_5": round(latest["SMA_5"], 2) if pd.notna(latest["SMA_5"]) else None,
        "latest_SMA_20": round(latest["SMA_20"], 2) if pd.notna(latest["SMA_20"]) else None,
        "latest_RSI_14": round(latest["RSI_14"], 2) if pd.notna(latest["RSI_14"]) else None,
        # Top moves
        "largest_single_day_gains_%": gains_list,
        "largest_single_day_losses_%": losses_list,
        # Figure paths
        "price_figure_path": PRICE_FIG,
        "volume_figure_path": VOLUME_FIG,
    }
    return summary


def set_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except ValueError:
        plt.style.use("seaborn-whitegrid")


def plot_price_with_sma(df: pd.DataFrame, filename: str) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df.index, df["Adj Close"], label="Adj Close", color="steelblue", linewidth=2)
    ax.plot(df.index, df["SMA_5"],  label="5-day SMA", color="orange", linewidth=1.5)
    ax.plot(df.index, df["SMA_20"], label="20-day SMA", color="green",  linewidth=1.5)
    ax.set_title("Nvidia (NVDA) Price and Moving Averages – Sep 23 to Oct 23 2025")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(filename, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved price/SMA figure → {filename}")


def plot_volume(df: pd.DataFrame, filename: str) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(df.index, df["Volume"] / 1_000_000, color="slategray")
    ax.set_title("Nvidia (NVDA) Daily Trading Volume – Sep 23 to Oct 23 2025")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume (Millions of Shares)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(filename, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved volume figure → {filename}")


# --------------------------------------------------------------------------
# 3. Main workflow
# --------------------------------------------------------------------------
def main() -> None:
    ensure_directories()

    df = fetch_stock_data(TICKER, START_DATE, END_DATE_EX)
    df.to_csv(RAW_CSV)
    logger.info(f"Raw data written to {RAW_CSV}")
    print(f"Raw data preview:\n{df.head()}\n")

    df = add_indicators(df)
    summary = compute_summary_metrics(df)

    # Console summary for the writer
    print("===== Nvidia (NVDA) Summary – Sep 23 to Oct 23 2025 =====")
    print(f"Period % Change (Adj Close): {summary['percentage_change_full_period_%']}%")
    print(f"Max Adj Close: ${summary['max_adj_close']}")
    print(f"Min Adj Close: ${summary['min_adj_close']}")
    print(f"Average Adj Close: ${summary['average_adj_close']}")
    print(f"Average Daily Volume: {summary['average_daily_volume']:,} shares\n")

    print("Latest indicators as of 2025-10-23:")
    print(f"  Adj Close: ${summary['latest_adj_close']}")
    print(f"  5-day SMA: ${summary['latest_SMA_5']}")
    print(f"  20-day SMA: ${summary['latest_SMA_20']}")
    print(f"  14-day RSI: {summary['latest_RSI_14']}\n")

    print("Top 3 Single-Day Gains (%):")
    for item in summary["largest_single_day_gains_%"]:
        print(f"  {item['date']}: {item['pct_change']}%")

    print("\nTop 3 Single-Day Losses (%):")
    for item in summary["largest_single_day_losses_%"]:
        print(f"  {item['date']}: {item['pct_change']}%")

    print("\nFigure files saved:")
    print(f"  Price/SMA chart → {summary['price_figure_path']}")
    print(f"  Volume chart    → {summary['volume_figure_path']}")
    print("=========================================================\n")

    # Charts
    plot_price_with_sma(df, PRICE_FIG)
    plot_volume(df, VOLUME_FIG)

    logger.info("NVDA monthly analysis complete.")


if __name__ == "__main__":
    main()