"""
Iteration: Outer 4, Inner 3
Timestamp: 20251026_151937
Execution Status: SUCCESS
Feedback from Planner:
No feedback yet
"""

# filename: generate_nvda_charts.py
"""
Generates 1-month performance & risk/return charts for Nvidia, peers, and
benchmarks.  Saves:
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
# Apply a seaborn-like dark-grid style bundled with Matplotlib
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except ValueError:
    plt.style.use("seaborn-darkgrid")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# -----------------------------------------------------------------------------
# Configuration & constants
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
    "NVDA": "#2E8B57",
    "AMD":  "#D62728",
    "INTC": "#1F77B4",
    "QCOM": "#9467BD"
}
COLOR_MAP_BENCH = {
    "NVDA": "#2E8B57",
    "SOXX": "#FF7F0E",
    "SPY":  "#7F7F7F"
}

TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 13
TICK_FONTSIZE  = 11


# -----------------------------------------------------------------------------
# Helpers
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
    Returns a DataFrame of Adjusted Close prices for the tickers list.
    Structure: index = dates, columns = ticker symbols.
    """
    try:
        logging.info(f"Fetching data for tickers: {tickers}")
        raw = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False    # keep raw + dividends (Adj Close provided)
            # 'group_by' left as default ('column') for easier handling
        )

        # The default MultiIndex columns -> ('Adj Close', <ticker>)
        if isinstance(raw.columns, pd.MultiIndex):
            if "Adj Close" not in raw.columns.levels[0]:
                raise KeyError("'Adj Close' not found in downloaded data.")
            adj_close = raw["Adj Close"].copy()
            adj_close.columns.name = None  # remove level name for cleanliness
        else:
            # Single-ticker case: raw has single-level columns
            if "Adj Close" not in raw.columns:
                raise KeyError("'Adj Close' not found in downloaded data.")
            adj_close = raw[["Adj Close"]].copy()
            adj_close.columns = tickers  # rename to ticker symbol

        # Drop any columns that are completely NA (e.g., failed downloads)
        adj_close = adj_close.dropna(axis=1, how="all")

        if adj_close.empty:
            raise ValueError("No valid Adjusted Close data retrieved.")

        missing = set(tickers) - set(adj_close.columns)
        if missing:
            logging.warning(f"No data for tickers: {list(missing)} (excluded)")

        logging.info("Data fetch successful.")
        return adj_close
    except Exception as exc:
        logging.exception("Data fetch failed.")
        raise


def calculate_metrics(price_df: pd.DataFrame) -> dict:
    """
    Calculate period return (%) and annualised volatility (%) per ticker.
    """
    metrics = {}
    daily_returns = price_df.pct_change().dropna()
    for tkr in price_df.columns:
        start_price = price_df[tkr].iloc[0]
        end_price   = price_df[tkr].iloc[-1]
        ret_pct     = (end_price / start_price - 1) * 100
        vol_pct     = daily_returns[tkr].std() * np.sqrt(252) * 100
        metrics[tkr] = {
            "Start_Adj_Close": round(float(start_price), 2),
            "End_Adj_Close":   round(float(end_price), 2),
            "Return_%":        round(float(ret_pct), 2),
            "Volatility_%":    round(float(vol_pct), 1)
        }
    return metrics


def plot_indexed_performance(price_df, tickers, color_map, title, filename):
    """
    Save an indexed-performance line chart.
    """
    logging.info(f"Creating {filename}")
    plt.figure(figsize=FIGSIZE, dpi=FIG_DPI)

    for tkr in tickers:
        if tkr not in price_df.columns:
            continue
        series    = price_df[tkr]
        indexed   = series / series.iloc[0] * 100
        lw        = 2.5 if tkr == "NVDA" else 1.5
        plt.plot(indexed.index, indexed.values,
                 label=tkr,
                 color=color_map[tkr],
                 linewidth=lw)

        # Annotate return %
        ret_pct = (series.iloc[-1] / series.iloc[0] - 1) * 100
        plt.text(indexed.index[-1], indexed.values[-1],
                 f"{ret_pct:+.1f}%",
                 color=color_map[tkr],
                 fontsize=10,
                 va="center",
                 ha="left")

    plt.title(title, fontsize=TITLE_FONTSIZE, pad=15)
    plt.xlabel("Date",  fontsize=LABEL_FONTSIZE)
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
    Create & save risk-return scatter plot.
    """
    logging.info(f"Creating {filename}")
    plt.figure(figsize=FIGSIZE, dpi=FIG_DPI)

    df = pd.DataFrame(metrics).T
    x = df["Volatility_%"].astype(float)
    y = df["Return_%"].astype(float)

    # Build marker sizes via market cap (optional)
    sizes = []
    for tkr in df.index:
        try:
            cap = yf.Ticker(tkr).info.get("marketCap")
            sizes.append(((cap or 0) / 1e9) ** 0.5 * 20 if cap else 80)
        except Exception:
            sizes.append(80)

    colors = [COLOR_MAP_COMP.get(t, COLOR_MAP_BENCH.get(t, "#333333")) for t in df.index]

    for i, tkr in enumerate(df.index):
        size = sizes[i] * (1.8 if tkr == "NVDA" else 1.0)
        plt.scatter(x[i], y[i], s=size, color=colors[i],
                    edgecolor="black", alpha=0.8, zorder=3)
        plt.text(x[i] + 0.5, y[i], tkr, fontsize=11, va="center")

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
# Main
# -----------------------------------------------------------------------------
def main():
    setup_logging()
    ensure_directories()

    # 1) Fetch data
    price_df = fetch_price_history(ALL_TICKERS, START_DATE, END_DATE)
    price_df.to_csv(DATA_PATH)
    logging.info(f"Data stored at {DATA_PATH}")
    print(f"Data saved to {DATA_PATH}")

    # 2) Metrics
    metrics = calculate_metrics(price_df)
    print("Latest Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 3) Visualisations
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

    # 4) Summary
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