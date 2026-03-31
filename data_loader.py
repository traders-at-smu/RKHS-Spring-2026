"""
Data Loader — bridges raw parquet/npz/pkl data into the 10-kernel pipeline.

Usage:
    from data_loader import DataLoader

    dl = DataLoader("data/")
    dl.load_all()

    # Fast kernels (dollar-bar resolution)
    lob_snapshots = dl.lob_snapshots("2023-07-10")
    lob_features  = dl.lob_features("2023-07-10")
    trades_df     = dl.trades()           # → VPIN, Kyle's Lambda, Hawkes
    ohlcv_df      = dl.ohlcv()            # → VRP, MacroMotion

    # Slow kernels (daily resolution)
    options_df    = dl.options()           # → GammaExposure, VannaCharm
    vrp_df        = dl.vrp_features()     # → VRP (pre-built daily features)

    # Kernel-ready outputs
    vpin_features = dl.build_vpin_inputs()
    kyle_inputs   = dl.build_kyle_inputs()
    hawkes_inputs = dl.build_hawkes_inputs()
    vrp_inputs    = dl.build_vrp_inputs()
"""

import os
import glob
import pickle
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ── LOB snapshot wrapper (matches LOB.py expected input) ─────────────────────

@dataclass
class LOBSnapshot:
    """Single limit-order-book snapshot for LOB kernel."""
    bid_prices: np.ndarray     # (n_levels,)
    bid_volumes: np.ndarray    # (n_levels,)
    ask_prices: np.ndarray     # (n_levels,)
    ask_volumes: np.ndarray    # (n_levels,)
    timestamp: int             # dollar-bar index


# ── Main loader ──────────────────────────────────────────────────────────────

class DataLoader:
    """
    Loads and caches all data sources for the Pristine-RKHS pipeline.

    Parameters
    ----------
    data_dir : str
        Path to the data/ folder (default: "data/").
    symbol : str
        Symbol prefix for file matching (default: "CL").
    """

    def __init__(self, data_dir: str = "data/", symbol: str = "CL"):
        self.data_dir = os.path.abspath(data_dir)
        self.symbol = symbol

        # Caches (lazy-loaded)
        self._trades: Optional["pd.DataFrame"] = None
        self._ohlcv: Optional["pd.DataFrame"] = None
        self._definitions: Optional["pd.DataFrame"] = None
        self._options: Optional["pd.DataFrame"] = None
        self._lob_features_cache: Dict[str, dict] = {}
        self._lob_snapshots_cache: Dict[str, list] = {}
        self._gram_cache: Dict[str, dict] = {}

    # ── File paths ───────────────────────────────────────────────────────

    @property
    def trades_path(self) -> str:
        return os.path.join(self.data_dir, f"trades_{self.symbol}_full.parquet")

    @property
    def ohlcv_path(self) -> str:
        return os.path.join(self.data_dir, f"ohlcv1m_{self.symbol}_full.parquet")

    @property
    def definitions_path(self) -> str:
        return os.path.join(self.data_dir, f"definition_{self.symbol}_FUT_full.parquet")

    @property
    def options_path(self) -> str:
        return os.path.join(self.data_dir, "statistics_LO_OPT_full.parquet")

    @property
    def lob_dir(self) -> str:
        return os.path.join(self.data_dir, "LOB")

    # ── Raw data loaders (lazy, cached) ──────────────────────────────────

    def trades(self, date_range: Optional[Tuple[str, str]] = None) -> "pd.DataFrame":
        """
        Load trade ticks.  Columns: ts_event, price, size, side, symbol, ...

        Used by: VPIN, Kyle's Lambda, Hawkes, LOB (dollar bars).
        """
        self._require_pandas()
        if self._trades is None:
            self._trades = pd.read_parquet(self.trades_path)
            if "ts_event" in self._trades.columns:
                self._trades["ts_event"] = pd.to_datetime(
                    self._trades["ts_event"], utc=True
                )
        df = self._trades
        if date_range:
            start, end = pd.Timestamp(date_range[0], tz="UTC"), pd.Timestamp(date_range[1], tz="UTC")
            df = df[(df["ts_event"] >= start) & (df["ts_event"] <= end)]
        return df

    def ohlcv(self, date_range: Optional[Tuple[str, str]] = None) -> "pd.DataFrame":
        """
        Load 1-minute OHLCV bars.  Columns: open, high, low, close, volume, symbol.
        Index: ts_event (datetime64[ns, UTC]).

        Used by: VRP (aggregated to daily), MacroMotion.
        """
        self._require_pandas()
        if self._ohlcv is None:
            self._ohlcv = pd.read_parquet(self.ohlcv_path)
        df = self._ohlcv
        if date_range:
            start, end = pd.Timestamp(date_range[0], tz="UTC"), pd.Timestamp(date_range[1], tz="UTC")
            df = df[(df.index >= start) & (df.index <= end)]
        return df

    def definitions(self) -> "pd.DataFrame":
        """Load contract definitions (expiration, multiplier, etc.)."""
        self._require_pandas()
        if self._definitions is None:
            self._definitions = pd.read_parquet(self.definitions_path)
        return self._definitions

    def options(self, date_range: Optional[Tuple[str, str]] = None) -> "pd.DataFrame":
        """
        Load options statistics.  Columns: ts_event, price, quantity,
        stat_type, symbol, ...

        Used by: GammaExposure, VannaCharm.
        Warning: ~168M rows, ~2.2 GB.  Use date_range to filter.
        """
        self._require_pandas()
        if self._options is None:
            self._options = pd.read_parquet(self.options_path)
            if "ts_event" in self._options.columns:
                self._options["ts_event"] = pd.to_datetime(
                    self._options["ts_event"], utc=True
                )
        df = self._options
        if date_range:
            start, end = pd.Timestamp(date_range[0], tz="UTC"), pd.Timestamp(date_range[1], tz="UTC")
            df = df[(df["ts_event"] >= start) & (df["ts_event"] <= end)]
        return df

    # ── LOB pre-computed data ────────────────────────────────────────────

    def available_lob_dates(self) -> List[str]:
        """Return sorted list of dates (YYYYMMDD) with LOB features."""
        pattern = os.path.join(
            self.lob_dir, f"lob_features_{self.symbol}", "*_lob_features.npz"
        )
        files = sorted(glob.glob(pattern))
        return [os.path.basename(f)[:8] for f in files]

    def lob_features(self, date_str: str) -> dict:
        """
        Load pre-computed LOB features for one day.

        Returns dict with keys:
            volume_profile : (n_bars, 20) — normalised volume at each level
            book_shape     : (n_bars, 20) — bid+ask depth profile
            depth_imbalance: (n_bars, 10) — level-by-level imbalance
            timestamps     : (n_bars,)    — UTC timestamp strings
            bar_indices    : (n_bars,)    — integer bar indices
        """
        if date_str not in self._lob_features_cache:
            path = os.path.join(
                self.lob_dir, f"lob_features_{self.symbol}",
                f"{date_str}_lob_features.npz",
            )
            data = dict(np.load(path, allow_pickle=True))
            self._lob_features_cache[date_str] = data
        return self._lob_features_cache[date_str]

    def lob_snapshots(self, date_str: str) -> list:
        """
        Load raw LOB snapshots for one day.

        Returns list of dicts, one per dollar bar, each containing
        bid/ask prices and volumes at multiple levels.
        """
        if date_str not in self._lob_snapshots_cache:
            path = os.path.join(
                self.lob_dir, f"lob_snapshots_{self.symbol}",
                f"{date_str}_lob_snapshots.pkl",
            )
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._lob_snapshots_cache[date_str] = data
        return self._lob_snapshots_cache[date_str]

    def lob_gram(self, date_str: str) -> dict:
        """
        Load pre-computed Gram matrix diagnostics for one day.

        Returns dict with keys:
            top_eig_ratios, eff_dims, n_windows, window_size,
            ell_vp, ell_bs, ell_di
        """
        if date_str not in self._gram_cache:
            path = os.path.join(
                self.lob_dir, f"gram_matrices_{self.symbol}",
                f"{date_str}_gram.npz",
            )
            data = dict(np.load(path, allow_pickle=True))
            self._gram_cache[date_str] = data
        return self._gram_cache[date_str]

    # ── Kernel-ready builders ────────────────────────────────────────────

    def build_dollar_bars(
        self,
        date_range: Optional[Tuple[str, str]] = None,
        dollar_threshold: float = 1_000_000.0,
    ) -> "pd.DataFrame":
        """
        Aggregate trade ticks into dollar bars.

        Returns DataFrame with columns:
            timestamp, open, high, low, close, volume, dollar_volume, n_trades
        """
        self._require_pandas()
        trades = self.trades(date_range)

        bars = []
        cum_dollars = 0.0
        bar_open = bar_high = bar_low = None
        bar_close = 0.0
        bar_vol = 0
        bar_n = 0
        bar_ts = None

        for _, row in trades.iterrows():
            p, s = row["price"], int(row["size"])
            dollars = p * s

            if bar_open is None:
                bar_open = p
                bar_high = p
                bar_low = p
                bar_ts = row["ts_event"]

            bar_high = max(bar_high, p)
            bar_low = min(bar_low, p)
            bar_close = p
            bar_vol += s
            bar_n += 1
            cum_dollars += dollars

            if cum_dollars >= dollar_threshold:
                bars.append({
                    "timestamp": bar_ts,
                    "open": bar_open,
                    "high": bar_high,
                    "low": bar_low,
                    "close": bar_close,
                    "volume": bar_vol,
                    "dollar_volume": cum_dollars,
                    "n_trades": bar_n,
                })
                cum_dollars = 0.0
                bar_open = None
                bar_vol = 0
                bar_n = 0

        return pd.DataFrame(bars)

    def build_vpin_inputs(
        self,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> "pd.DataFrame":
        """
        Prepare trade data for VPIN kernel.

        Returns DataFrame with columns: timestamp, price, size
        (matches VPIN.py run_pipeline() expected input).
        """
        self._require_pandas()
        trades = self.trades(date_range)
        return trades[["ts_event", "price", "size"]].rename(
            columns={"ts_event": "timestamp"}
        )

    def build_kyle_inputs(
        self,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> "pd.DataFrame":
        """
        Prepare trade data for Kyle's Lambda kernel.

        Returns DataFrame with columns: price, size, signed_size
        (matches Kyles_Lambda.py expected input).
        """
        self._require_pandas()
        trades = self.trades(date_range)
        df = trades[["price", "size", "side"]].copy()
        df["signed_size"] = df["size"].astype(float) * df["side"].map(
            {"A": 1.0, "B": -1.0, "N": 0.0}
        ).fillna(0.0)
        return df[["price", "size", "signed_size"]]

    def build_hawkes_inputs(
        self,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> np.ndarray:
        """
        Extract event timestamps for Hawkes kernel.

        Returns 1-D array of trade times in seconds since epoch.
        """
        trades = self.trades(date_range)
        ts = pd.to_datetime(trades["ts_event"], utc=True)
        return (ts.astype(np.int64) / 1e9).values

    def build_daily_ohlcv(
        self,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> "pd.DataFrame":
        """
        Aggregate 1-minute bars into daily OHLCV.

        Returns DataFrame indexed by date with columns:
            open, high, low, close, volume
        """
        self._require_pandas()
        ohlcv = self.ohlcv(date_range)
        daily = ohlcv.groupby(ohlcv.index.date).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        daily.index = pd.to_datetime(daily.index)
        daily.index.name = "date"
        return daily

    def build_vrp_inputs(
        self,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> "pd.DataFrame":
        """
        Build daily feature DataFrame for VRP kernel.

        Computes realised volatility from 1-min bars. Automatically merges
        VIX and put/call ratio from data/external/ if available (run
        fetch_external_data.py first).

        Returns DataFrame with columns:
            date, close, rv_30d, iv_30d, iv_90d, vix, put_call_ratio
        """
        self._require_pandas()
        daily = self.build_daily_ohlcv(date_range)

        log_ret = np.log(daily["close"] / daily["close"].shift(1))
        daily["rv_30d"] = log_ret.rolling(30).std() * np.sqrt(252)

        # Try to load external data
        ext = os.path.join(self.data_dir, "external")

        # VIX → iv_30d proxy and vix column
        vix_path = os.path.join(ext, "vix_daily.parquet")
        if os.path.exists(vix_path):
            vix = pd.read_parquet(vix_path)
            vix.index = pd.to_datetime(vix.index)
            daily = daily.join(vix[["vix"]], how="left")
            daily["iv_30d"] = daily["vix"] / 100.0  # VIX is annualised %
            daily["iv_90d"] = daily["iv_30d"]        # approximate
        else:
            daily["iv_30d"] = np.nan
            daily["iv_90d"] = np.nan
            daily["vix"] = np.nan

        # Put/call ratio
        pcr_path = os.path.join(ext, "put_call_ratio.parquet")
        if os.path.exists(pcr_path):
            pcr = pd.read_parquet(pcr_path)
            pcr.index = pd.to_datetime(pcr.index)
            daily = daily.join(pcr[["put_call_ratio"]], how="left")
        else:
            daily["put_call_ratio"] = np.nan

        daily = daily.reset_index()
        return daily[["date", "close", "rv_30d", "iv_30d", "iv_90d",
                       "vix", "put_call_ratio"]].dropna(subset=["rv_30d"])

    def build_event_calendar(self) -> "pd.DataFrame":
        """
        Load the economic event calendar for EventProximity kernel.

        Returns DataFrame with columns: event_date, event_type, magnitude
        """
        self._require_pandas()
        path = os.path.join(self.data_dir, "external", "event_calendar.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Event calendar not found at {path}. "
                "Run fetch_external_data.py first."
            )
        return pd.read_parquet(path)

    def build_macro_inputs(
        self,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> "pd.DataFrame":
        """
        Build daily close prices for MacroMotion kernel.

        Automatically merges SPY and XLE from data/external/ if available.

        Returns DataFrame with columns: date, close, spy_close, xle_close
        """
        self._require_pandas()
        daily = self.build_daily_ohlcv(date_range)

        ext = os.path.join(self.data_dir, "external")

        # SPY
        spy_path = os.path.join(ext, "spy_daily.parquet")
        if os.path.exists(spy_path):
            spy = pd.read_parquet(spy_path)
            spy.index = pd.to_datetime(spy.index)
            daily = daily.join(spy[["close"]].rename(columns={"close": "spy_close"}), how="left")
        else:
            daily["spy_close"] = np.nan

        # XLE (energy sector)
        xle_path = os.path.join(ext, "xle_daily.parquet")
        if os.path.exists(xle_path):
            xle = pd.read_parquet(xle_path)
            xle.index = pd.to_datetime(xle.index)
            daily = daily.join(xle[["close"]].rename(columns={"close": "xle_close"}), how="left")
        else:
            daily["xle_close"] = np.nan

        return daily[["close", "spy_close", "xle_close"]].reset_index()

    def build_lob_snapshot_objects(
        self,
        date_str: str,
        n_levels: int = 5,
    ) -> List[LOBSnapshot]:
        """
        Convert raw LOB snapshots into LOBSnapshot objects
        that LOB.py's RKHSLayer.gram_matrix() expects.
        """
        raw = self.lob_snapshots(date_str)
        snapshots = []
        for i, snap in enumerate(raw):
            if isinstance(snap, dict):
                snapshots.append(LOBSnapshot(
                    bid_prices=np.array(snap.get("bid_prices", snap.get("bids", []))[:n_levels], dtype=float),
                    bid_volumes=np.array(snap.get("bid_volumes", snap.get("bid_sizes", []))[:n_levels], dtype=float),
                    ask_prices=np.array(snap.get("ask_prices", snap.get("asks", []))[:n_levels], dtype=float),
                    ask_volumes=np.array(snap.get("ask_volumes", snap.get("ask_sizes", []))[:n_levels], dtype=float),
                    timestamp=i,
                ))
        return snapshots

    # ── Date range helpers ───────────────────────────────────────────────

    def date_range(self) -> Tuple[str, str]:
        """Return (first_date, last_date) from the LOB feature files."""
        dates = self.available_lob_dates()
        if not dates:
            raise ValueError("No LOB feature files found")
        return dates[0], dates[-1]

    def trading_dates(self) -> List[str]:
        """Return sorted list of trading dates available in LOB data."""
        return self.available_lob_dates()

    # ── Utility ──────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Print a summary of available data."""
        lines = [f"DataLoader — {self.data_dir}", ""]

        for name, path in [
            ("Trades", self.trades_path),
            ("OHLCV 1m", self.ohlcv_path),
            ("Definitions", self.definitions_path),
            ("Options", self.options_path),
        ]:
            exists = os.path.exists(path)
            size = os.path.getsize(path) / 1e6 if exists else 0
            lines.append(f"  {name:15s} {'OK' if exists else 'MISSING':8s} {size:>8.1f} MB")

        lob_dates = self.available_lob_dates()
        lines.append(f"\n  LOB dates:       {len(lob_dates)} days")
        if lob_dates:
            lines.append(f"    range: {lob_dates[0]} — {lob_dates[-1]}")

        return "\n".join(lines)

    def load_all(self):
        """Pre-load trades and OHLCV into cache (options loaded on demand)."""
        self._require_pandas()
        print("Loading trades...")
        self.trades()
        print("Loading OHLCV...")
        self.ohlcv()
        print("Done.")

    @staticmethod
    def _require_pandas():
        if not HAS_PANDAS:
            raise ImportError(
                "pandas is required. Install with: pip install pandas pyarrow"
            )


# ── CLI quick-check ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/"
    dl = DataLoader(data_dir)
    print(dl.summary())
