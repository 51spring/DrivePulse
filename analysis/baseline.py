"""Context-baseline anomaly detection for hard-braking events.

Input is a *traversals* table: one row per vehicle pass through one road segment.

    trip_id, driver_id, segment_id, date (YYYY-MM-DD), hour (0-23), hard_brakes (int)

Counting events per traversal makes the denominator (traffic) explicit, so we compare
*rates*, not raw counts.

Two detectors share one baseline:

1. Segment anomalies: "segment S, on date D, in time bucket B had far more hard brakes
   than its usual rate for that bucket" (incident, debris, construction, ...).
2. Driver anomalies: "driver K brakes hard more often than other drivers did on the
   same segments at the same times" (standardized incidence ratio, SIR). Cells flagged
   as segment anomalies are excluded first, so road problems are not blamed on drivers.

Baseline (empirical Bayes, leave-one-date-out):
    prior(s, b)  = segment rate(s) x bucket factor(b)
    lambda(s, b) = (Y_hist + k * prior) / (E_hist + k)
where Y_hist/E_hist are hard brakes/traversals of (s, b) on *other* dates and k is a
pseudo-exposure that pulls sparse cells toward the prior.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

# (first hour, last hour, name)
HOUR_BANDS = [(0, 6, "night"), (7, 9, "am_peak"), (10, 16, "midday"), (17, 19, "pm_peak"), (20, 23, "evening")]
_BAND_OF_HOUR = np.empty(24, dtype=object)
for _lo, _hi, _name in HOUR_BANDS:
    _BAND_OF_HOUR[_lo:_hi + 1] = _name

REQUIRED_COLUMNS = ["trip_id", "driver_id", "segment_id", "date", "hour", "hard_brakes"]


def slot_label(hour, minute, slot_minutes: int):
    """'07:30'-style label of the slot containing hour:minute."""
    start = (np.asarray(hour, dtype=int) * 60 + np.asarray(minute, dtype=int)) // slot_minutes * slot_minutes
    return pd.Series(start // 60).astype(str).str.zfill(2) + ":" + pd.Series(start % 60).astype(str).str.zfill(2)


def add_context(df: pd.DataFrame, slot_minutes: int | None = None) -> pd.DataFrame:
    """Adds day_type, band, parent_bucket (= day_type_band) and bucket columns.

    slot_minutes=None: bucket = day_type_band (5 bands per day).
    slot_minutes=30  : bucket = day_type_HH:MM (needs a 'minute' column); the band bucket
                       becomes the parent level the slot baseline is shrunk toward.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    weekday = pd.to_datetime(out["date"]).dt.dayofweek
    out["day_type"] = np.where(weekday >= 5, "weekend", "weekday")
    out["band"] = _BAND_OF_HOUR[out["hour"].astype(int).to_numpy()]
    out["parent_bucket"] = out["day_type"] + "_" + out["band"]
    if slot_minutes:
        if "minute" not in out.columns:
            raise ValueError("slot_minutes needs a 'minute' column")
        out["bucket"] = out["day_type"] + "_" + slot_label(out["hour"], out["minute"], slot_minutes).to_numpy()
    else:
        out["bucket"] = out["parent_bucket"]
    out["hard_brakes"] = out["hard_brakes"].astype(int)
    return out


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def poisson_upper_p(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """P(X >= observed) for X ~ Poisson(expected)."""
    observed = np.asarray(observed)
    expected = np.maximum(np.asarray(expected, dtype=float), 1e-12)
    return np.where(observed <= 0, 1.0, poisson.sf(observed - 1, expected))


def cell_baseline(df: pd.DataFrame, k: float = 30.0, k_segment: float = 200.0) -> pd.DataFrame:
    """Per (segment, bucket, date) cell: exposure, count and leave-date-out expected rate.

    Returns columns: segment_id, parent_bucket, bucket, date, exposure, count, base_rate, expected.

    Band mode (bucket == parent_bucket):
        prior = segment rate x bucket factor
    Slot mode (e.g. 30-min buckets inside a band):
        parent = (segment x band rate, shrunk to segment rate x band factor)
        prior  = parent x slot factor (how the slot compares with its band network-wide)
    """
    if "parent_bucket" not in df.columns:
        df = df.assign(parent_bucket=df["bucket"])
    cells = (
        df.groupby(["segment_id", "parent_bucket", "bucket", "date"], as_index=False)
        .agg(exposure=("hard_brakes", "size"), count=("hard_brakes", "sum"))
    )
    return cell_rates(cells, k=k, k_segment=k_segment)


def cell_rates(cells: pd.DataFrame, k: float = 30.0, k_segment: float = 200.0) -> pd.DataFrame:
    """cell_baseline on cells that are already aggregated (segment_id, parent_bucket, bucket,
    date, exposure, count), e.g. counted chunk by chunk from a traversals file too big for memory."""
    cells = cells.copy()
    # Totals used for leave-one-date-out.
    sb = cells.groupby(["segment_id", "bucket"])[["exposure", "count"]].transform("sum")
    seg = cells.groupby("segment_id")[["exposure", "count"]].transform("sum")
    bucket_tot = cells.groupby("bucket")[["exposure", "count"]].transform("sum")
    g_exp, g_cnt = cells["exposure"].sum(), cells["count"].sum()

    e_sb = sb["exposure"] - cells["exposure"]
    y_sb = sb["count"] - cells["count"]
    e_seg = seg["exposure"] - cells["exposure"]
    y_seg = seg["count"] - cells["count"]

    global_rate = g_cnt / g_exp
    bucket_factor = (bucket_tot["count"] / bucket_tot["exposure"]) / global_rate
    seg_rate = (y_seg + k_segment * global_rate) / (e_seg + k_segment)

    if (cells["parent_bucket"] != cells["bucket"]).any():
        cols = ["exposure", "count"]
        sp_all = cells.groupby(["segment_id", "parent_bucket"])[cols].transform("sum")
        sp_day = cells.groupby(["segment_id", "parent_bucket", "date"])[cols].transform("sum")
        e_sp = sp_all["exposure"] - sp_day["exposure"]   # leave the whole date out
        y_sp = sp_all["count"] - sp_day["count"]
        parent_tot = cells.groupby("parent_bucket")[cols].transform("sum")
        parent_global = parent_tot["count"] / parent_tot["exposure"]
        parent_rate = (y_sp + k * seg_rate * parent_global / global_rate) / (e_sp + k)
        slot_factor = (bucket_tot["count"] / bucket_tot["exposure"]) / parent_global
        prior = parent_rate * slot_factor
    else:
        prior = seg_rate * bucket_factor
    cells["base_rate"] = (y_sb + k * prior) / (e_sb + k)
    cells["expected"] = cells["base_rate"] * cells["exposure"]
    return cells


def detect_segment_anomalies(
    df: pd.DataFrame,
    *,
    k: float = 30.0,
    min_exposure: int = 10,
    min_count: int = 3,
    min_ratio: float = 2.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Flags (segment, bucket, date) cells whose hard-brake count is far above baseline."""
    cells = cell_baseline(df, k=k)
    cells["rate"] = cells["count"] / cells["exposure"]
    cells["rate_ratio"] = cells["count"] / cells["expected"].clip(lower=1e-12)
    cells["p_value"] = poisson_upper_p(cells["count"].to_numpy(), cells["expected"].to_numpy())

    tested = cells["exposure"] >= min_exposure
    cells["q_value"] = 1.0
    cells.loc[tested, "q_value"] = benjamini_hochberg(cells.loc[tested, "p_value"].to_numpy())
    cells["flagged"] = (
        tested
        & (cells["q_value"] < alpha)
        & (cells["count"] >= min_count)
        & (cells["rate_ratio"] >= min_ratio)
    )
    return cells.sort_values("p_value").reset_index(drop=True)


def detect_window_spikes(
    df: pd.DataFrame,
    band_cells: pd.DataFrame,
    *,
    bin_minutes: int = 10,
    max_bins: int = 3,
    min_exposure: int = 10,
    min_count: int = 5,
    min_ratio: float = 1.5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Scan for short spikes inside each (segment, band, date) cell.

    A 20-minute incident on a busy road is diluted when the whole band is counted at
    once. Here the cell is cut into `bin_minutes` bins and every contiguous window of
    1..max_bins bins is tested against the baseline:

        expected(window) = base_rate(segment, band, leave-date-out)
                           x bin factor (network-wide rate of that bin / rate of the band)
                           x traversals in the window

    The smallest Poisson p-value is Bonferroni-corrected for the number of windows tried,
    then Benjamini-Hochberg is applied across cells.

    df: add_context(..., slot_minutes=None) output with a 'minute' column.
    band_cells: detect_segment_anomalies(df) in band mode (provides base_rate).
    Returns one row per band cell with its best window.
    """
    if "minute" not in df.columns:
        raise ValueError("window scan needs a 'minute' column")
    d = df.assign(bin=(df["hour"].astype(int) * 60 + df["minute"].astype(int)) // bin_minutes)
    keys = ["segment_id", "bucket", "date"]

    bin_rate = d.groupby(["day_type", "bin"])["hard_brakes"].mean().rename("bin_rate")
    band_rate = d.groupby("bucket")["hard_brakes"].mean().rename("band_rate")
    bins = (
        d.groupby(keys + ["day_type", "bin"], as_index=False)
        .agg(exposure=("hard_brakes", "size"), count=("hard_brakes", "sum"))
        .merge(band_cells[keys + ["base_rate"]], on=keys, how="left")
        .join(bin_rate, on=["day_type", "bin"])
        .join(band_rate, on="bucket")
    )
    bins["mu"] = bins["base_rate"] * bins["bin_rate"] / bins["band_rate"].clip(lower=1e-12) * bins["exposure"]

    rows = []
    for key, g in bins.groupby(keys, sort=False):
        b0, b1 = int(g["bin"].min()), int(g["bin"].max())
        y = np.zeros(b1 - b0 + 1)
        mu = np.zeros(b1 - b0 + 1)
        y[g["bin"].to_numpy() - b0] = g["count"].to_numpy()
        mu[g["bin"].to_numpy() - b0] = g["mu"].to_numpy()
        cy, cm = np.concatenate([[0], np.cumsum(y)]), np.concatenate([[0], np.cumsum(mu)])
        best, n_windows = (1.0, 0, 1, 0.0, 0.0), 0
        for length in range(1, min(max_bins, len(y)) + 1):
            wy = cy[length:] - cy[:-length]
            wm = cm[length:] - cm[:-length]
            p = poisson_upper_p(wy, wm)
            n_windows += len(p)
            i = int(np.argmin(p))
            if p[i] < best[0]:
                best = (p[i], i, length, wy[i], wm[i])
        p_min, start, length, wy, wm = best
        start_min = (b0 + start) * bin_minutes
        rows.append((*key, int(g["exposure"].sum()), f"{start_min // 60:02d}:{start_min % 60:02d}",
                     length * bin_minutes, wy, wm, min(1.0, p_min * n_windows)))

    out = pd.DataFrame(rows, columns=keys + ["exposure", "window_start", "window_minutes",
                                             "w_count", "w_expected", "p_value"])
    out["w_ratio"] = out["w_count"] / out["w_expected"].clip(lower=1e-12)
    tested = out["exposure"] >= min_exposure
    out["q_value"] = 1.0
    out.loc[tested, "q_value"] = benjamini_hochberg(out.loc[tested, "p_value"].to_numpy())
    out["flagged"] = tested & (out["q_value"] < alpha) & (out["w_count"] >= min_count) & (out["w_ratio"] >= min_ratio)
    return out.sort_values("p_value").reset_index(drop=True)


def detect_driver_anomalies(
    df: pd.DataFrame,
    segment_cells: pd.DataFrame,
    *,
    shrink: float = 5.0,
    min_sir: float = 1.5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compares each driver's hard brakes with what other drivers did in the same context.

    segment_cells: output of detect_segment_anomalies (provides base_rate and flags).
    """
    keys = ["segment_id", "bucket", "date"]
    t = df.merge(segment_cells[keys + ["base_rate", "flagged"]], on=keys, how="left")
    t = t[~t["flagged"].fillna(False).astype(bool)]  # road anomalies are not the driver's fault

    drivers = t.groupby("driver_id").agg(
        traversals=("hard_brakes", "size"),
        observed=("hard_brakes", "sum"),
        expected=("base_rate", "sum"),
    )
    drivers["naive_rate_per100"] = 100 * drivers["observed"] / drivers["traversals"]
    drivers["sir"] = drivers["observed"] / drivers["expected"].clip(lower=1e-12)
    drivers["sir_shrunk"] = (drivers["observed"] + shrink) / (drivers["expected"] + shrink)
    drivers["p_value"] = poisson_upper_p(drivers["observed"].to_numpy(), drivers["expected"].to_numpy())
    drivers["q_value"] = benjamini_hochberg(drivers["p_value"].to_numpy())
    drivers["flagged"] = (drivers["q_value"] < alpha) & (drivers["sir"] >= min_sir)
    return drivers.sort_values("sir_shrunk", ascending=False).reset_index()
