"""
Fetch external data sources needed by the slow kernels.

Downloads:
  1. VIX index              → VRP kernel (iv proxy, vix)
  2. SPY daily closes        → MacroMotion kernel (benchmark)
  3. XLE daily closes        → MacroMotion kernel (energy sector)
  4. Put/call ratio          → VRP kernel (derived from options parquet)
  5. Economic event calendar → EventProximity kernel (FOMC, CPI, NFP, etc.)

Usage:
    python fetch_external_data.py [--data-dir data/]

Outputs saved to data/external/:
    vix_daily.parquet
    spy_daily.parquet
    xle_daily.parquet
    put_call_ratio.parquet
    event_calendar.parquet
"""

import os
import sys
import argparse
from datetime import date

import numpy as np
import pandas as pd


# ── Date range (matches LOB data: 2023-07-02 to 2024-12-31) ─────────────────

START = "2023-07-01"
END   = "2025-01-02"  # pad by 1 day for yfinance end-exclusive


# ═════════════════════════════════════════════════════════════════════════════
# 1–3. Market data via yfinance
# ═════════════════════════════════════════════════════════════════════════════

def fetch_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance."""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, auto_adjust=True)
    df.index = df.index.tz_localize(None)  # strip tz for consistency
    df.index.name = "date"
    return df


def fetch_vix(out_dir: str):
    """VIX index → proxy for 30-day implied vol."""
    print("  Fetching VIX...")
    df = fetch_yfinance("^VIX", START, END)
    df = df[["Close"]].rename(columns={"Close": "vix"})
    path = os.path.join(out_dir, "vix_daily.parquet")
    df.to_parquet(path)
    print(f"    {len(df)} rows → {path}")
    return df


def fetch_spy(out_dir: str):
    """SPY daily closes → MacroMotion benchmark."""
    print("  Fetching SPY...")
    df = fetch_yfinance("SPY", START, END)
    df = df[["Close", "Volume"]].rename(
        columns={"Close": "close", "Volume": "volume"}
    )
    path = os.path.join(out_dir, "spy_daily.parquet")
    df.to_parquet(path)
    print(f"    {len(df)} rows → {path}")
    return df


def fetch_xle(out_dir: str):
    """XLE (energy sector ETF) → MacroMotion sector proxy."""
    print("  Fetching XLE...")
    df = fetch_yfinance("XLE", START, END)
    df = df[["Close", "Volume"]].rename(
        columns={"Close": "close", "Volume": "volume"}
    )
    path = os.path.join(out_dir, "xle_daily.parquet")
    df.to_parquet(path)
    print(f"    {len(df)} rows → {path}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 4. Put/call ratio from options parquet
# ═════════════════════════════════════════════════════════════════════════════

def derive_put_call_ratio(data_dir: str, out_dir: str):
    """
    Derive daily put/call volume ratio from statistics_LO_OPT_full.parquet.

    Uses stat_type and symbol to classify puts vs calls, counts daily volume.
    """
    print("  Deriving put/call ratio from options data...")
    opts_path = os.path.join(data_dir, "statistics_LO_OPT_full.parquet")
    df = pd.read_parquet(opts_path, columns=["ts_event", "symbol", "quantity"])

    # Classify put vs call from symbol (e.g., "LOZ3 P5550" vs "LOZ3 C4800")
    df["option_type"] = df["symbol"].str.extract(r" ([PC])\d+$", expand=False)
    df = df.dropna(subset=["option_type"])

    # Daily date
    df["date"] = pd.to_datetime(df["ts_event"], utc=True).dt.date

    # Count rows per day per type as proxy for activity
    daily = df.groupby(["date", "option_type"]).size().unstack(fill_value=0)
    daily.columns = ["call_count", "put_count"] if "C" in daily.columns else daily.columns

    if "C" in daily.columns and "P" in daily.columns:
        daily = daily.rename(columns={"C": "call_count", "P": "put_count"})
    daily["put_call_ratio"] = daily["put_count"] / daily["call_count"].replace(0, np.nan)

    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"

    path = os.path.join(out_dir, "put_call_ratio.parquet")
    daily.to_parquet(path)
    print(f"    {len(daily)} rows → {path}")
    del df
    return daily


# ═════════════════════════════════════════════════════════════════════════════
# 5. Economic event calendar (static)
# ═════════════════════════════════════════════════════════════════════════════

def build_event_calendar(out_dir: str):
    """
    Build a static calendar of market-moving events for EventProximity kernel.

    Covers: FOMC decisions, CPI releases, NFP releases, quarterly OpEx,
    EIA crude inventory reports.
    """
    print("  Building economic event calendar...")

    events = []

    # ── FOMC meeting dates (2023 H2 + 2024) ─────────────────────────────
    fomc_dates = [
        # 2023
        "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
        # 2024
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    ]
    for d in fomc_dates:
        events.append({"event_date": d, "event_type": "FOMC", "magnitude": 1.0})

    # ── CPI releases (approx. 2nd week of each month) ───────────────────
    cpi_dates = [
        # 2023
        "2023-07-12", "2023-08-10", "2023-09-13", "2023-10-12",
        "2023-11-14", "2023-12-12",
        # 2024
        "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10",
        "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
        "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    ]
    for d in cpi_dates:
        events.append({"event_date": d, "event_type": "CPI", "magnitude": 0.8})

    # ── Non-Farm Payrolls (first Friday of each month) ───────────────────
    nfp_dates = [
        # 2023
        "2023-07-07", "2023-08-04", "2023-09-01", "2023-10-06",
        "2023-11-03", "2023-12-08",
        # 2024
        "2024-01-05", "2024-02-02", "2024-03-08", "2024-04-05",
        "2024-05-03", "2024-06-07", "2024-07-05", "2024-08-02",
        "2024-09-06", "2024-10-04", "2024-11-01", "2024-12-06",
    ]
    for d in nfp_dates:
        events.append({"event_date": d, "event_type": "NFP", "magnitude": 0.7})

    # ── Quarterly OpEx (3rd Friday of March, June, Sep, Dec) ─────────────
    opex_dates = [
        "2023-09-15", "2023-12-15",
        "2024-03-15", "2024-06-21", "2024-09-20", "2024-12-20",
    ]
    for d in opex_dates:
        events.append({"event_date": d, "event_type": "OpEx", "magnitude": 0.6})

    # ── EIA Weekly Crude Inventory (Wednesdays, sample every ~month) ─────
    eia_dates = [
        "2023-07-12", "2023-08-09", "2023-09-06", "2023-10-04",
        "2023-11-01", "2023-12-06",
        "2024-01-03", "2024-02-07", "2024-03-06", "2024-04-03",
        "2024-05-01", "2024-06-05", "2024-07-03", "2024-08-07",
        "2024-09-04", "2024-10-02", "2024-11-06", "2024-12-04",
    ]
    for d in eia_dates:
        events.append({"event_date": d, "event_type": "EIA_Crude", "magnitude": 0.5})

    df = pd.DataFrame(events)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values("event_date").reset_index(drop=True)

    path = os.path.join(out_dir, "event_calendar.parquet")
    df.to_parquet(path)
    print(f"    {len(df)} events → {path}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Fetch external data for RKHS pipeline")
    parser.add_argument("--data-dir", default="data/", help="Path to data/ folder")
    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, "external")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Output: {out_dir}\n")

    # 1-3: Yahoo Finance
    vix = fetch_vix(out_dir)
    spy = fetch_spy(out_dir)
    xle = fetch_xle(out_dir)

    # 4: Put/call ratio from options
    pcr = derive_put_call_ratio(args.data_dir, out_dir)

    # 5: Event calendar
    cal = build_event_calendar(out_dir)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Done. Files saved to data/external/:")
    print(f"  vix_daily.parquet        {len(vix):>5} rows")
    print(f"  spy_daily.parquet        {len(spy):>5} rows")
    print(f"  xle_daily.parquet        {len(xle):>5} rows")
    print(f"  put_call_ratio.parquet   {len(pcr):>5} rows")
    print(f"  event_calendar.parquet   {len(cal):>5} events")
    print("=" * 60)

    # ── Wire into VRP ────────────────────────────────────────────────────
    print("\nTo complete VRP inputs, the data_loader can now merge these:")
    print("  dl.build_vrp_inputs() will use vix + put_call_ratio + rv_30d")
    print("\nSentiment kernel still needs a news API (newsapi.org or Benzinga).")
    print("  Sign up → set NEWSAPI_KEY env var → run a separate headline fetch.")


if __name__ == "__main__":
    main()
