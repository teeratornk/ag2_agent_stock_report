"""
Iteration: Outer 3, Inner 2
Timestamp: 20251026_140529
Execution Status: FAILED
Feedback from Planner:
REVIEW_COMPLETE
"""

"""
File: nvda_competitor_analysis.py
Description: Fetches 1-month price & volume data for Nvidia (NVDA), its key competitors
             (AMD, INTC, QCOM) and benchmarks (SPY, SOXX).  Calculates basic metrics
             and produces two line charts:
                 1) NVDA vs competitors (indexed performance)
                 2) NVDA vs benchmarks (kept from previous iteration)
Outputs:
    • data/semis_1mo_raw.csv      – All tickers’ raw price/volume data for the period
    • figures/nvda_vs_competitors.png – New competitor comparison chart
    • figures/nvda_vs_benchmarks.png  – Existing benchmark chart (re-created, unchanged)
Returns (via print): metrics dict & figure paths
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend – MUST be first
import os
import logging
from datetime import datetime
from typing import Dict, List

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------- config & logging ----------------------------- #
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = 'data'
FIG_DIR = 'figures'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

DATE_START = '2025-09-23'
DATE_END = '2025-10-23'

# Core tickers
TARGET = ['NVDA']
COMPETITORS = ['AMD', 'INTC', 'QCOM']
BENCHMARKS = ['SPY', 'SOXX']
ALL_TICKERS = TARGET + COMPETITORS + BENCHMARKS


# ------------------------------ helpers ------------------------------------ #
def fetch_price_history(tickers: List[str],
                        start: str,
                        end: str) -> pd.DataFrame:
    """
    Download adjusted close and volume for a list of tickers.
    Returns a tidy DataFrame with MultiIndex (Date, Ticker).
    """
    try:
        logging.info(f"Fetching data for tickers: {tickers}")
        data = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            progress=False,
            group_by='ticker',
            auto_adjust=False,
            threads=True
        )
        if data.empty:
            raise ValueError("No data fetched – please check tickers or date range.")

        # yfinance returns multi-level columns if multiple tickers
        tidy_frames = []
        for ticker in tickers:
            df = data[ticker][['Adj Close', 'Volume']].copy()
            df.columns = ['Adj_Close', 'Volume']
            df['Ticker'] = ticker
            tidy_frames.append(df)

        tidy_df = pd.concat(tidy_frames)
        tidy_df.index.names = ['Date']
        tidy_df.reset_index(inplace=True)
        tidy_df.set_index(['Date', 'Ticker'], inplace=True)
        return tidy_df

    except Exception as e:
        logging.exception("Error fetching price history")
        raise e


def calculate_metrics(tidy_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute required metrics for each ticker.
    """
    metrics: Dict[str, Dict[str, float]] = {}
    for ticker in tidy_df.index.get_level_values('Ticker').unique():
        df_t = tidy_df.xs(ticker, level='Ticker')
        df_t = df_t.sort_index()
        start_price = df_t['Adj_Close'].iloc[0]
        end_price = df_t['Adj_Close'].iloc[-1]
        period_return = (end_price - start_price) / start_price * 100

        # Daily % change for volatility
        daily_pct = df_t['Adj_Close'].pct_change().dropna()
        if not daily_pct.empty:
            ann_vol = daily_pct.std(ddof=0) * np.sqrt(252) * 100
        else:
            ann_vol = np.nan

        metrics[ticker] = {
            'Start_Adj_Close': round(start_price, 2),
            'End_Adj_Close': round(end_price, 2),
            'Period_Return_%': round(period_return, 2),
            'Volatility_%': round(ann_vol, 1)
        }

    return metrics


def plot_indexed_performance(tidy_df: pd.DataFrame,
                             tickers: List[str],
                             title: str,
                             filename: str) -> None:
    """
    Plot indexed performance (start=100) for given tickers and save to file.
    """
    try:
        plt.figure(figsize=(12, 6))
        for ticker in tickers:
            df_t = tidy_df.xs(ticker, level='Ticker').sort_index()
            base_price = df_t['Adj_Close'].iloc[0]
            indexed = df_t['Adj_Close'] / base_price * 100
            plt.plot(df_t.index, indexed, label=ticker)

        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Indexed Price (Start = 100)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        save_path = os.path.join(FIG_DIR, filename)
        plt.savefig(save_path, dpi=110, bbox_inches='tight')
        plt.close()
        logging.info(f"Figure saved: {save_path}")
    except Exception as e:
        logging.exception("Failed to create chart: %s", filename)
        raise e


# ------------------------------ main flow ---------------------------------- #
def main() -> None:
    # 1. Fetch data
    tidy_df = fetch_price_history(ALL_TICKERS, DATE_START, DATE_END)

    # 2. Save raw data
    raw_csv_path = os.path.join(DATA_DIR, 'semis_1mo_raw.csv')
    tidy_df.to_csv(raw_csv_path)
    logging.info(f"Raw data saved: {raw_csv_path}")
    print(f'Data saved to {raw_csv_path}')

    # 3. Calculate metrics
    metrics = calculate_metrics(tidy_df)

    # 4. Build figure paths dict (existing + new)
    fig_paths = {
        'Benchmarks': os.path.join(FIG_DIR, 'nvda_vs_benchmarks.png'),
        'Competitors': os.path.join(FIG_DIR, 'nvda_vs_competitors.png'),
        # Previous figures not recreated here but can be added if needed
    }

    # 5. Charts
    # 5a. Competitor chart
    plot_indexed_performance(
        tidy_df,
        tickers=TARGET + COMPETITORS,  # NVDA + competitors
        title='Nvidia vs. Key Semiconductor Competitors – 1-Month Performance',
        filename='nvda_vs_competitors.png'
    )

    # 5b. Benchmark chart (identical logic as previous iteration)
    plot_indexed_performance(
        tidy_df,
        tickers=TARGET + BENCHMARKS,    # NVDA + benchmarks
        title='Nvidia vs. Benchmarks – 1-Month Performance',
        filename='nvda_vs_benchmarks.png'
    )

    # 6. Output summary for reviewer
    print("\n=== Metrics Summary ===")
    for tkr, vals in metrics.items():
        print(f"{tkr}: {vals}")

    print("\n=== Figure Paths ===")
    for key, path in fig_paths.items():
        print(f"{key}: {path}")

    # Note: In an integrated pipeline we would return these objects
    # (metrics, fig_paths) to the next agent. For now we just print.
    return metrics, fig_paths


# ----------------------------- script start -------------------------------- #
if __name__ == '__main__':
    main()