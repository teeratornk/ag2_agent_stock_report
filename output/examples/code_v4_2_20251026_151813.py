"""
Iteration: Outer 4, Inner 2
Timestamp: 20251026_151813
Execution Status: FAILED
Feedback from Planner:
No feedback yet
"""

# filename: generate_nvda_charts.py
"""
Generates 1-month performance & risk/return charts for Nvidia, its semiconductor
peers, and market benchmarks—without requiring seaborn.  
Figures and data are saved to relative folders:
  • data/semis_1mo_raw.csv
  • figures/nvda_vs_competitors.png
  • figures/nvda_vs_benchmarks.png
  • figures/semis_risk_return.png
"""

import os
import logging
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
# Apply a seaborn-like darkgrid style that ships with Matplotlib
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except ValueError:
    # Fallback for older Matplotlib versions
    plt.style.use("seaborn-darkgrid")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
FIG_DPI = 110
FIGSIZE = (12, 7)

COMPETITOR_TICKERS = ["NVDA", "AMD", "INTC", "QCOM"]
BENCHMARK_TICKERS  = ["NVDA", "SOXX", "SPY"]
ALL_TICKERS        = sorted(set(COMPETITOR_TICKERS + BENCHMARK_TICKERS))

START_DATE = (datetime(2025, 10, 23) - timedelta(days=30)).strftime("%Y-%m-%d")   # 2025-09-23
END_DATE   = "2025-10-23"

DATA_PATH    = "data/semis_1mo_raw.csv"
FIG_PATH_COMPETITORS = "figures/nvda_vs_competitors.png"
FIG_PATH_BENCHMARKS  = "figures/nvda_vs_benchmarks.png"
FIG_PATH_RISK_RET    = "figures/semis_risk_return.png"

COLOR_MAP_COMP = {
    "NVDA": "#2E8B57",  # green
    "AMD":  "#D62728",  # red
    "INTC": "#1F77B4",  # blue
    "QCOM": "#9467BD"   # purple
}
COLOR_MAP_BENCH = {
    "NVDA": "#2E8B57",  # green
    "SOXX": "#FF7F0E",  # orange
    "SPY":  "#7F7F7F"   # gray
}

TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 13
TICK_FONTSIZE  = 11


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s | %(message)s",
        level=logging.INFO
    )


def ensure_directories() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)


def fetch_price_history(tickers, start_date, end_date) -> pd.DataFrame:
    """
    Fetch Adjusted Close prices for the given tickers.
    """
    try:
        logging.info(f"Fetching data for tickers: {tickers}")
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            progress=False,
            group_by="ticker",
            auto_adjust=False
        )
        if len(tickers) == 1:
            data = data["Adj Close"].to_frame(name=tickers[0])
        else:
            data = data["Adj Close"]
        logging.info("Data fetch successful.")
        return data
    except Exception as exc:
        logging.exception("Error fetching price data.")
        raise


def calculate_metrics(price_df):
    """
    Compute period return % and annualised volatility % for each ticker.
    """
    metrics = {}
    daily_returns = price_df.pct_change().dropna()
    for ticker in price_df.columns:
        start_price = price_df[ticker].iloc[0]
        end_price   = price_df[ticker].iloc[-1]
        period_ret  = (end_price / start_price - 1) * 100
        vol_annual  = daily_returns[ticker].std() * np.sqrt(252) * 100
        metrics[ticker] = {
            "Start_Adj_Close": round(float(start_price), 2),
            "End_Adj_Close":   round(float(end_price), 2),
            "Return_%":        round(float(period_ret), 2),
            "Volatility_%":    round(float(vol_annual), 1)
        }
    return metrics


def plot_indexed_performance(price_df: pd.DataFrame,
                             tickers,
                             color_map,
                             title: str,
                             filename: str) -> None:
    """
    Plot indexed performance (Base 100) for provided tickers.
    """
    logging.info(f"Generating {filename}")
    plt.figure(figsize=FIGSIZE, dpi=FIG_DPI)

    for ticker in tickers:
        series = price_df[ticker]
        indexed = series / series.iloc[0] * 100
        linewidth = 2.5 if ticker == "NVDA" else 1.5
        plt.plot(indexed.index, indexed.values,
                 label=ticker,
                 color=color_map[ticker],
                 linewidth=linewidth)

        ret_pct = (series.iloc[-1] / series.iloc[0] - 1) * 100
        plt.text(indexed.index[-1], indexed.values[-1],
                 f"{ret_pct:+.1f}%",
                 color=color_map[ticker],
                 fontsize=10,
                 va="center",
                 ha="left")

    plt.title(title, fontsize=TITLE_FONTSIZE, pad=15)
    plt.xlabel("Date", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Indexed Price (Base 100)", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE, rotation=45)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.legend(bbox_to_anchor=(1.15, 0.5), loc="center left", frameon=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved {filename}")


def plot_risk_return(metrics: dict, filename: str) -> None:
    """
    Scatter plot of Annualised Volatility (%) vs Return (%).
    """
    logging.info(f"Generating {filename}")
    plt.figure(figsize=FIGSIZE, dpi=FIG_DPI)

    df = pd.DataFrame(metrics).T
    x = df["Volatility_%"]
    y = df["Return_%"]

    # Derive marker sizes via market cap where possible
    sizes = []
    for ticker in df.index:
        try:
            cap = yf.Ticker(ticker).info.get("marketCap")
            if cap and cap > 0:
                sizes.append((cap / 1e9) ** 0.5 * 20)  # Scaled bubble size
            else:
                sizes.append(80)
        except Exception:
            sizes.append(80)

    colors = [COLOR_MAP_COMP.get(t, COLOR_MAP_BENCH.get(t, "#333333")) for t in df.index]

    for i, ticker in enumerate(df.index):
        size = sizes[i] * (1.8 if ticker == "NVDA" else 1.0)
        plt.scatter(x[i], y[i],
                    s=size,
                    color=colors[i],
                    edgecolor="black",
                    alpha=0.8,
                    zorder=3)
        plt.text(x[i] + 0.5, y[i],
                 ticker,
                 fontsize=11,
                 va="center")

    # Average lines
    plt.axvline(x.mean(), color="grey", linestyle="--", linewidth=1)
    plt.axhline(y.mean(), color="grey", linestyle="--", linewidth=1)

    plt.title("Risk-Return Snapshot (Past Month)", fontsize=TITLE_FONTSIZE, pad=15)
    plt.xlabel("Annualised Volatility (%)", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Return (%)", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved {filename}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    setup_logging()
    ensure_directories()

    # Fetch data
    price_df = fetch_price_history(ALL_TICKERS, START_DATE, END_DATE)
    price_df.to_csv(DATA_PATH)
    logging.info(f"Raw data saved to {DATA_PATH}")
    print(f"Data saved to {DATA_PATH}")

    # Calculate metrics
    metrics = calculate_metrics(price_df)
    print("Latest Metrics:")
    for ticker, vals in metrics.items():
        print(f"{ticker}: {vals}")

    # Generate visualisations
    plot_indexed_performance(price_df[COMPETITOR_TICKERS],
                             COMPETITOR_TICKERS,
                             COLOR_MAP_COMP,
                             "1-Month Indexed Performance: Nvidia vs. Semiconductor Peers",
                             FIG_PATH_COMPETITORS)

    plot_indexed_performance(price_df[BENCHMARK_TICKERS],
                             BENCHMARK_TICKERS,
                             COLOR_MAP_BENCH,
                             "1-Month Indexed Performance: Nvidia vs. Market Benchmarks",
                             FIG_PATH_BENCHMARKS)

    plot_risk_return(metrics, FIG_PATH_RISK_RET)

    # Summary
    print("\nFigures generated and saved:")
    print(f" - {FIG_PATH_COMPETITORS}")
    print(f" - {FIG_PATH_BENCHMARKS}")
    print(f" - {FIG_PATH_RISK_RET}")

    return {
        "metrics": metrics,
        "figures": {
            "competitors": FIG_PATH_COMPETITORS,
            "benchmarks":  FIG_PATH_BENCHMARKS,
            "risk_return": FIG_PATH_RISK_RET
        }
    }


if __name__ == "__main__":
    main()