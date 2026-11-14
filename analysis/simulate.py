"""Synthetic multi-driver traversal data with known ground truth.

The point is to test whether the context baseline separates three things that an
absolute threshold (e.g. "hard brakes per 100 km") mixes up:

  * congested roads / rush hours  -> many hard brakes, nobody's fault
  * incidents (segment x date x bucket spikes) -> road anomaly
  * genuinely risky drivers       -> driver anomaly

Road: a line of segments seg_000..seg_NNN. Segments 0-9 form a congested corridor with a
3x base rate. 35% of drivers commute through that corridor; 10% of drivers are risky
(2.5-3.5x). The two groups are independent, so a driver's raw rate is confounded by route.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from baseline import add_context

BAND_MULT = {"night": 0.6, "am_peak": 1.8, "midday": 1.0, "pm_peak": 2.0, "evening": 0.9}
WEEKEND_MULT = 0.8
CORRIDOR = range(0, 10)


def simulate(
    n_segments: int = 40,
    n_drivers: int = 600,
    n_days: int = 56,
    start: str = "2026-09-01",
    n_incidents: int = 25,
    incident_mult: float = 5.0,
    risky_share: float = 0.10,
    corridor_share: float = 0.35,
    seed: int = 7,
):
    """Returns (traversals, truth_incidents, truth_drivers)."""
    rng = np.random.default_rng(seed)

    seg_rate = np.clip(rng.lognormal(np.log(0.012), 0.5, n_segments), 0.003, 0.05)
    seg_rate[list(CORRIDOR)] *= 3.0

    risky = rng.random(n_drivers) < risky_share
    risk_mult = np.where(risky, rng.uniform(2.5, 3.5, n_drivers), rng.lognormal(0, 0.15, n_drivers))
    corridor = rng.random(n_drivers) < corridor_share

    # Each driver's commute: a contiguous run of segments.
    commute = []
    for d in range(n_drivers):
        length = int(rng.integers(5, 10))
        if corridor[d]:
            s0 = int(rng.integers(0, 4))
        else:
            s0 = int(rng.integers(10, n_segments - length))
        commute.append(np.arange(s0, s0 + length))

    dates = pd.date_range(start, periods=n_days, freq="D")
    rows_driver, rows_seg, rows_day, rows_hour, rows_trip = [], [], [], [], []
    trip_id = 0

    def add_trip(d: int, day: int, hour: int, segs: np.ndarray):
        nonlocal trip_id
        n = len(segs)
        rows_driver.append(np.full(n, d))
        rows_seg.append(segs)
        rows_day.append(np.full(n, day))
        rows_hour.append(np.full(n, hour))
        rows_trip.append(np.full(n, trip_id))
        trip_id += 1

    def random_route() -> np.ndarray:
        length = int(rng.integers(3, 9))
        s0 = int(rng.integers(0, n_segments - length))
        segs = np.arange(s0, s0 + length)
        return segs if rng.random() < 0.5 else segs[::-1]

    for day, date in enumerate(dates):
        weekend = date.dayofweek >= 5
        for d in range(n_drivers):
            if not weekend:
                if rng.random() < 0.85:
                    add_trip(d, day, int(rng.integers(7, 10)), commute[d])
                if rng.random() < 0.85:
                    add_trip(d, day, int(rng.integers(17, 20)), commute[d][::-1])
                for _ in range(rng.poisson(0.4)):
                    add_trip(d, day, int(rng.choice([10, 11, 13, 14, 15, 20, 21])), random_route())
            else:
                for _ in range(rng.poisson(1.0)):
                    add_trip(d, day, int(rng.integers(9, 23)), random_route())

    t = pd.DataFrame({
        "trip_id": np.concatenate(rows_trip),
        "driver_id": np.concatenate(rows_driver),
        "segment_idx": np.concatenate(rows_seg),
        "day": np.concatenate(rows_day),
        "hour": np.concatenate(rows_hour),
    })
    t["segment_id"] = "seg_" + t["segment_idx"].astype(str).str.zfill(3)
    t["date"] = dates[t["day"].to_numpy()].strftime("%Y-%m-%d")
    t["driver_id"] = "drv_" + t["driver_id"].astype(str).str.zfill(4)
    t["trip_id"] = "trip_" + t["trip_id"].astype(str)
    t["hard_brakes"] = 0
    t = add_context(t)

    # Incidents: pick busy (segment, bucket, date) cells.
    cells = t.groupby(["segment_id", "bucket", "date"]).size().rename("exposure").reset_index()
    busy = cells[cells["exposure"] >= 30]
    incidents = busy.sample(n=min(n_incidents, len(busy)), random_state=seed)[["segment_id", "bucket", "date"]]
    incident_key = set(map(tuple, incidents.to_numpy()))
    is_incident = np.fromiter(
        ((s, b, dt) in incident_key for s, b, dt in zip(t["segment_id"], t["bucket"], t["date"])),
        dtype=bool, count=len(t),
    )

    driver_idx = t["driver_id"].str[4:].astype(int).to_numpy()
    rate = (
        seg_rate[t["segment_idx"].to_numpy()]
        * t["band"].map(BAND_MULT).to_numpy()
        * np.where(t["day_type"] == "weekend", WEEKEND_MULT, 1.0)
        * risk_mult[driver_idx]
        * np.where(is_incident, incident_mult, 1.0)
    )
    t["hard_brakes"] = rng.poisson(rate)

    truth_drivers = pd.DataFrame({
        "driver_id": [f"drv_{d:04d}" for d in range(n_drivers)],
        "risky": risky,
        "risk_mult": risk_mult,
        "corridor_commuter": corridor,
    })
    cols = ["trip_id", "driver_id", "segment_id", "date", "hour", "hard_brakes"]
    return t[cols], incidents.reset_index(drop=True), truth_drivers
