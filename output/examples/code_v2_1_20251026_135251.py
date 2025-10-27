"""
Iteration: Outer 2, Inner 1
Timestamp: 20251026_135251
Execution Status: FAILED
Feedback from Planner:
No feedback yet
"""

"""
Engineer Script – Iteration 2  
Adds benchmark comparison (SPY, SOXX) to existing Nvidia (NVDA) analysis.

Directory structure (relative paths – we are already in the `coding` folder):
    data/      → raw CSV files
    figures/   → png charts

Outputs:
    • Updated metrics dict printed to console (JSON-serialisable)
    • CSV for each ticker in data/
    • Three figures:
        - figures/nvda_price_moving_avg.png      (already required)
        - figures/nvda_volume.png                (already required)
        - figures/nvda_vs_benchmarks.png         (NEW – indexed performance)

Environment constraints honoured:
    • matplotlib non-interactive backend (‘Agg’)
    • No seaborn / GUI / plt.show()
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import matplotlib
matplotlib.use('Agg')  # non-interactive backend

import os
import json
import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Logging configuration
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
# Helper Functions
# --------------------------------------------------------------------------- #
def ensure_directories() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR,  exist_ok=True)
    logging.info("Ensured data/ and figures/ directories exist.")

def fetch_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical price data for a single ticker within [start, end].
    Falls back to 'Close' if 'Adj Close' missing. Ensures required columns.
    """
    try:
        logging.info(f"Fetching data for {ticker} …")
        data = (
            yf.Ticker(ticker)
            .history(start=start, end=end, auto_adjust=False)
            .loc[:, ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        )
    except Exception as e:
        logging.error(f"yfinance failed for {ticker}: {e}")
        raise

    # Fallback logic for 'Adj Close'
    if "Adj Close" not in data.columns or data["Adj Close"].isna().all():
        logging.warning(f"'Adj Close' missing for {ticker}, duplicating 'Close'.")
        data["Adj Close"] = data["Close"]

    # Save raw data
    csv_path = f"{DATA_DIR}/{ticker.lower()}_{START_DATE}_to_{END_DATE}.csv"
    data.to_csv(csv_path)
    logging.info(f"Saved raw data to {csv_path}")

    return data


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI) for a price series.
    """
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def annualised_volatility(daily_returns: pd.Series) -> float:
    """
    Annualise the standard deviation of daily percentage returns.
    """
    return float(daily_returns.std(ddof=0) * np.sqrt(252) * 100)


def pct_return(start: float, end: float) -> float:
    """
    Compute percentage return rounded to 2 decimals.
    """
    return round((end / start - 1) * 100, 2)


# --------------------------------------------------------------------------- #
# Analysis Class (Single Responsibility)
# --------------------------------------------------------------------------- #
class StockAnalyzer:
    """
    Handles all computations for NVDA (and partially for benchmarks).
    """

    def __init__(self, df: pd.DataFrame, symbol: str):
        self.df = df.copy()
        self.symbol = symbol
        self.metrics: Dict = {}

    def calculate_basic_metrics(self) -> None:
        start_price = float(self.df["Adj Close"].iloc[0])
        end_price   = float(self.df["Adj Close"].iloc[-1])

        # Daily % change for volatility calculation
        pct_changes = self.df["Adj Close"].pct_change().dropna()

        self.metrics.update(
            Start_Adj_Close=round(start_price, 2),
            End_Adj_Close=round(end_price, 2),
            Period_Return_%=pct_return(start_price, end_price),
            Volatility_%=round(annualised_volatility(pct_changes), 1),
        )

    # ---------------- NVDA-specific calculations ---------------- #
    def calculate_technical_indicators(self) -> None:
        """
        NVDA specific: SMA_5, SMA_20, RSI_14, High/Low/Avg, Avg Volume,
        Top gains/losses.
        """
        # Simple Moving Averages
        self.df["SMA_5"] = self.df["Adj Close"].rolling(window=5).mean()
        self.df["SMA_20"] = self.df["Adj Close"].rolling(window=20).mean()

        # RSI
        self.df["RSI_14"] = compute_rsi(self.df["Adj Close"])

        high_price = float(self.df["High"].max())
        low_price  = float(self.df["Low"].min())
        avg_price  = float(self.df["Adj Close"].mean())

        average_volume = int(self.df["Volume"].mean())

        # Daily % change list (drop first NA)
        daily_pct = self.df["Adj Close"].pct_change().dropna() * 100
        daily_pct_sorted = daily_pct.sort_values(ascending=False)

        top_gains = (
            daily_pct_sorted.head(3)
            .round(2)
            .to_frame(name="pct_change")
            .reset_index()
            .rename(columns={"index": "date"})
        )
        top_losses = (
            daily_pct_sorted.tail(3)
            .round(2)
            .to_frame(name="pct_change")
            .reset_index()
            .rename(columns={"index": "date"})
            .sort_values(by="pct_change")  # losses already negative
        )

        # Build indicator dict
        self.metrics.update(
            High_Low_Avg={
                "High": round(high_price, 2),
                "Low": round(low_price, 2),
                "Average": round(avg_price, 2),
            },
            Average_Volume=average_volume,
            Latest_Indicators={
                "SMA_5": round(float(self.df["SMA_5"].iloc[-1]), 2),
                "SMA_20": round(float(self.df["SMA_20"].iloc[-1]), 2),
                "RSI_14": round(float(self.df["RSI_14"].iloc[-1]), 2),
            },
            Top_Gains_%=top_gains.to_dict(orient="records"),
            Top_Losses_%=top_losses.to_dict(orient="records"),
        )

    # ---------------- Figures ---------------- #
    def create_price_and_volume_charts(self) -> Dict[str, str]:
        """
        Produces:
            1. Price with SMA_5 & SMA_20
            2. Volume chart
        Returns dict of figure paths.
        """
        fig_paths = {}

        # 1. Price & SMA
        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df["Adj Close"], label="Adj Close", color="#333333")
        plt.plot(self.df.index, self.df["SMA_5"], label="SMA 5", color="#1f77b4")
        plt.plot(self.df.index, self.df["SMA_20"], label="SMA 20", color="#ff7f0e")
        plt.title("NVIDIA (NVDA) Price & Moving Averages – 1 Month")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        price_fig = f"{FIG_DIR}/nvda_price_moving_avg.png"
        plt.savefig(price_fig, dpi=120, bbox_inches="tight")
        plt.close()
        fig_paths["Price_SMA"] = price_fig
        logging.info(f"Saved price/SMA figure → {price_fig}")

        # 2. Volume
        plt.figure(figsize=(12, 4))
        plt.bar(self.df.index, self.df["Volume"], color="#76B900")
        plt.title("NVIDIA (NVDA) Trading Volume – 1 Month")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.grid(True, axis='y', alpha=0.3)
        volume_fig = f"{FIG_DIR}/nvda_volume.png"
        plt.savefig(volume_fig, dpi=120, bbox_inches="tight")
        plt.close()
        fig_paths["Volume"] = volume_fig
        logging.info(f"Saved volume figure → {volume_fig}")

        return fig_paths

# --------------------------------------------------------------------------- #
# Benchmark Figure
# --------------------------------------------------------------------------- #
def create_benchmark_chart(adj_close_df: pd.DataFrame) -> str:
    """
    Creates an indexed performance line chart for NVDA, SPY, SOXX.
    """
    indexed = adj_close_df / adj_close_df.iloc[0] * 100

    plt.figure(figsize=(12, 6))
    for ticker in indexed.columns:
        plt.plot(indexed.index, indexed[ticker], label=ticker,
                 color=COLOR_MAP.get(ticker, None), linewidth=2)

    plt.title("Nvidia vs Benchmarks – 1-Month Indexed Performance (Start = 100)")
    plt.xlabel("Date")
    plt.ylabel("Indexed Price (Start = 100)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    bench_fig = f"{FIG_DIR}/nvda_vs_benchmarks.png"
    plt.savefig(bench_fig, dpi=120, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved benchmark comparison figure → {bench_fig}")

    return bench_fig

# --------------------------------------------------------------------------- #
# Main Execution
# --------------------------------------------------------------------------- #
def main() -> None:
    ensure_directories()

    # 1. Fetch data for all tickers
    price_dfs = {}
    for ticker in TICKERS:
        price_dfs[ticker] = fetch_price_data(ticker, START_DATE, END_DATE)

    # 2. Combine Adj Close for benchmark figure
    adj_close_combined = pd.concat(
        {t: df["Adj Close"] for t, df in price_dfs.items()},
        axis=1
    )
    adj_close_combined.columns = TICKERS  # assure column order

    # 3. Analyze each ticker
    metrics_output: Dict[str, Dict] = {}
    for ticker in TICKERS:
        analyzer = StockAnalyzer(price_dfs[ticker], ticker)
        analyzer.calculate_basic_metrics()
        # NVDA needs extra
        if ticker == "NVDA":
            analyzer.calculate_technical_indicators()
            fig_paths = analyzer.create_price_and_volume_charts()
        metrics_output[ticker] = analyzer.metrics

    # 4. Create benchmark comparison figure
    bench_fig_path = create_benchmark_chart(adj_close_combined)

    # 5. Merge figure paths into output
    metrics_output.setdefault("Figure_Paths", {})
    # Add existing path dict if present (from NVDA calculation)
    metrics_output["Figure_Paths"].update(fig_paths)
    metrics_output["Figure_Paths"]["Benchmarks"] = bench_fig_path

    # 6. Print JSON-serialisable summary
    print(json.dumps(metrics_output, indent=2, default=str))
    logging.info("Analysis complete. Metrics dict printed above.")

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()