"""Reproducible EDA and volatility estimates for the DGM paper's 20 US stocks.

The frozen historical data end on 2017-11-10. MS and TSN are backfilled
before 2005 using a separate Yahoo archive, with an overlap-price scale match.
Run: python stock_eda_2000_2017.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TICKERS = [
    "IBM", "AMZN", "TIF", "AMGN", "BAC", "GIS", "CSCO", "KO", "CMCSA", "DE",
    "GE", "HD", "JNJ", "MS", "MSFT", "JWN", "PFE", "QCOM", "SBUX", "TSN",
]
START, END = "2000-01-01", "2017-12-31"
WINDOW = 60
TRADING_DAYS = 252

# Frozen public historical snapshots; TIF and JWN remain accessible here.
STOCKS_URL = (
    "https://raw.githubusercontent.com/shikha-aggarwal/machine_learning/"
    "55e3b5b8b47ddd9e93fd3808ac3fff6c85e3b7ce/"
    "datasets/stock_data/Stocks/{}.us.txt"
)
BACKFILL_URL = (
    "https://raw.githubusercontent.com/QuantSoftware/QuantSoftwareToolkit/"
    "0eb2c7a776c259a087fdcac1d3ff883eb0b5516c/"
    "QSTK/QSData/Yahoo/{}.csv"
)
DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
FIG_DIR = Path("results")


def download_csv(url, cache_file):
    if cache_file.exists():
        return pd.read_csv(cache_file)
    try:
        frame = pd.read_csv(url)
    except Exception as exc:
        raise RuntimeError(f"Cannot download {url}. Check your connection.") from exc
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_file, index=False)
    return frame


def price_series(frame, ticker, column):
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Date", column])
    frame = frame.loc[frame[column] > 0]
    series = (frame.drop_duplicates("Date", keep="last")
              .set_index("Date")[column].sort_index().loc[START:END])
    if series.empty:
        raise ValueError(f"No valid historical prices for {ticker}")
    return series.rename(ticker)


def get_prices(ticker):
    raw = download_csv(
        STOCKS_URL.format(ticker.lower()), CACHE_DIR / f"archive_{ticker}.csv"
    )
    current = price_series(raw, ticker, "Close")

    # The primary archive starts MS and TSN in 2005; recover 2000-2005.
    if ticker in {"MS", "TSN"}:
        old_raw = download_csv(
            BACKFILL_URL.format(ticker), CACHE_DIR / f"backfill_{ticker}.csv"
        )
        old = price_series(old_raw, ticker, "Adj Close")
        overlap = old.index.intersection(current.index)
        if overlap.empty:
            raise ValueError(f"No overlap to validate/backfill {ticker}")
        anchor = overlap[0]
        factor = current.loc[anchor] / old.loc[anchor]
        earlier = old.loc[old.index < current.index.min()] * factor
        current = pd.concat([earlier, current]).sort_index()
    return current


def plot_normalized_prices(zscores):
    fig, axes = plt.subplots(5, 4, figsize=(16, 13), sharex=True)
    sampled = zscores.iloc[::5]
    for ax, ticker in zip(axes.flat, TICKERS):
        ax.plot(sampled.index, sampled[ticker], linewidth=0.75)
        ax.axhline(0, linewidth=0.5, color="gray")
        ax.set_title(ticker, fontsize=11, weight="bold")
        ax.grid(alpha=0.18)
        ax.tick_params(axis="x", labelrotation=35, labelsize=7)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("20 stocks | 60-trading-day rolling price z-scores", fontsize=17)
    fig.supxlabel("Trading date")
    fig.supylabel("Rolling z-score")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG_DIR / "prices_60d_zscore.svg", bbox_inches="tight")
    plt.close(fig)


def plot_volatility(vol):
    fig, ax = plt.subplots(figsize=(10, 8))
    ordered = (100 * vol).sort_values()
    ax.barh(ordered.index, ordered.values)
    ax.set_xlabel("Annualized volatility (%), from daily log returns")
    ax.set_title("20-stock annualized volatility | 2000–2017")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "annualized_volatility.svg", bbox_inches="tight")
    plt.close(fig)


def plot_matrix(matrix, filename, title, correlation=False):
    values = matrix.to_numpy()
    fig, ax = plt.subplots(figsize=(12, 10))
    bound = 1.0 if correlation else max(0.01, np.abs(values).max())
    image = ax.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound)
    ax.set_xticks(range(len(TICKERS)), labels=TICKERS, rotation=65, ha="right")
    ax.set_yticks(range(len(TICKERS)), labels=TICKERS)
    for i in range(len(TICKERS)):
        for j in range(len(TICKERS)):
            value = values[i, j]
            label = f"{value:.2f}" if correlation else f"{value:.3f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=6,
                    color="white" if abs(value) > 0.60 * bound else "black")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    ax.set_title(title, fontsize=14, pad=16)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    prices = pd.concat([get_prices(t) for t in TICKERS], axis=1).sort_index()
    prices = prices.loc[START:END, TICKERS]

    coverage = pd.DataFrame({
        "first_date": prices.apply(lambda s: s.first_valid_index().date()),
        "last_date": prices.apply(lambda s: s.last_valid_index().date()),
        "valid_prices": prices.notna().sum(),
        "missing_prices": prices.isna().sum(),
    })
    coverage.index.name = "ticker"
    coverage.to_csv(DATA_DIR / "data_quality.csv")

    # Calculate returns on original adjusted prices, NOT on rolling z-scores.
    # Missing dates are not forward-filled, and incomplete return rows are removed.
    log_returns = np.log(prices).diff().dropna(how="any")
    if len(log_returns) < 1000:
        raise ValueError("Too few common-date returns for reliable estimation.")

    volatility = log_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    correlation = log_returns.corr()
    covariance = log_returns.cov() * TRADING_DAYS

    np.savetxt(DATA_DIR / "volatilities_2000_2017.csv",
               volatility.to_numpy(), delimiter=",", fmt="%.10f")
    np.savetxt(DATA_DIR / "correlation_matrix_2000_2017.csv",
               correlation.to_numpy(), delimiter=",", fmt="%.10f")
    np.savetxt(DATA_DIR / "annualized_covariance_matrix_2000_2017.csv",
               covariance.to_numpy(), delimiter=",", fmt="%.10f")
    pd.DataFrame({"ticker": TICKERS, "annualized_volatility": volatility.values,
                  "mean_daily_log_return": log_returns.mean().values,
                  "observations": len(log_returns)}).to_csv(
                      DATA_DIR / "eda_summary.csv", index=False)

    # A trailing window uses today and the previous 59 observations only.
    mean60 = prices.rolling(WINDOW, min_periods=WINDOW).mean()
    std60 = prices.rolling(WINDOW, min_periods=WINDOW).std(ddof=0)
    zscores = ((prices - mean60) / std60.replace(0, np.nan))
    plot_normalized_prices(zscores)
    plot_volatility(volatility)
    plot_matrix(correlation, "correlation_matrix.svg",
                "Daily log-return correlation | 2000–2017", True)
    plot_matrix(covariance, "annualized_covariance_matrix.svg",
                "Annualized covariance (volatility) matrix | 2000–2017")

    print(f"20 tickers, {len(log_returns):,} common daily returns, "
          f"{log_returns.index.min().date()} to {log_returns.index.max().date()}")
    print(f"Mean annualized volatility: {volatility.mean():.2%}")
    print(f"Saved CSV matrices to {DATA_DIR}/ and four SVG figures to {FIG_DIR}/")
    print("Ticker order:", ", ".join(TICKERS))


if __name__ == "__main__":
    main()
