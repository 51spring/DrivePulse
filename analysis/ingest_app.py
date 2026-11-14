"""Converts the in-car app's logs into the traversals table used by baseline.py.

App logs (pulled from the car / emulator):
    <session>_samples.csv : elapsed_ms, wall_time, speed_kmh, ..., geohash[, segment_id]  (10 Hz)
    hard_brakes.csv       : session_id, start_time, ..., segment_id, geohash, ...  (current app)
    events.csv            : session_id, start_time, ..., geohash, ...              (older app)

The app normally uploads traversals to the server itself; this script rebuilds them from
the raw logs. A traversal is a run of consecutive samples on the same road segment (the
map-matched segment_id; geohash cell for older logs or off-map driving) while the car is
moving. Each hard brake is assigned to the traversal of the same session whose time range
contains the event start.

Usage:
    python ingest_app.py LOG_DIR --driver-id car01 --out traversals.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MIN_MOVING_KMH = 5.0
MATCH_TOLERANCE = pd.Timedelta(seconds=2)


def samples_to_traversals(samples: pd.DataFrame, session_id: str, driver_id: str) -> pd.DataFrame:
    s = samples.copy()
    seg = s["segment_id"] if "segment_id" in s.columns else pd.Series(pd.NA, index=s.index)
    s["seg"] = seg.where(seg.notna() & (seg.astype(str).str.len() > 0), s["geohash"])
    s = s.dropna(subset=["seg"])
    s = s[s["seg"].astype(str).str.len() > 0]
    if s.empty:
        return pd.DataFrame()
    s["wall_time"] = pd.to_datetime(s["wall_time"], format="ISO8601")
    run_id = (s["seg"] != s["seg"].shift()).cumsum()
    runs = s.groupby(run_id).agg(
        segment_id=("seg", "first"),
        enter=("wall_time", "min"),
        exit=("wall_time", "max"),
        max_kmh=("speed_kmh", "max"),
        samples=("speed_kmh", "size"),
    )
    runs = runs[runs["max_kmh"] >= MIN_MOVING_KMH].reset_index(drop=True)
    runs["trip_id"] = session_id
    runs["driver_id"] = driver_id
    runs["date"] = runs["enter"].dt.strftime("%Y-%m-%d")
    runs["hour"] = runs["enter"].dt.hour
    runs["minute"] = runs["enter"].dt.minute
    return runs


def assign_events(traversals: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    counts = pd.Series(0, index=traversals.index)
    if events.empty:
        return counts
    for _, e in events.iterrows():
        cand = traversals[
            (traversals["trip_id"] == e["session_id"])
            & (traversals["enter"] - MATCH_TOLERANCE <= e["start_time"])
            & (e["start_time"] <= traversals["exit"] + MATCH_TOLERANCE)
        ]
        if cand.empty:
            continue
        where = e["segment_id"] if isinstance(e.get("segment_id"), str) and e["segment_id"] else e.get("geohash")
        same_cell = cand[cand["segment_id"] == where]
        counts[(same_cell if not same_cell.empty else cand).index[0]] += 1
    return counts


def ingest(log_dir: Path, driver_id: str) -> pd.DataFrame:
    parts = []
    for f in sorted(log_dir.glob("*_samples.csv")):
        session_id = f.name[: -len("_samples.csv")]
        samples = pd.read_csv(f, dtype={"geohash": str, "segment_id": str}, encoding="utf-8-sig")
        parts.append(samples_to_traversals(samples, session_id, driver_id))
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["trip_id", "driver_id", "segment_id", "date", "hour", "minute", "hard_brakes"])
    trav = pd.concat(parts, ignore_index=True)

    events = pd.concat([
        pd.read_csv(f, dtype={"geohash": str, "segment_id": str}, encoding="utf-8-sig")
        for f in (log_dir / "hard_brakes.csv", log_dir / "events.csv") if f.exists()
    ], ignore_index=True) if any((log_dir / n).exists() for n in ("hard_brakes.csv", "events.csv")) else pd.DataFrame()
    if not events.empty:
        events["start_time"] = pd.to_datetime(events["start_time"], format="ISO8601")
    trav["hard_brakes"] = assign_events(trav, events)
    return trav[["trip_id", "driver_id", "segment_id", "date", "hour", "minute", "hard_brakes", "enter", "exit", "max_kmh"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir", type=Path)
    ap.add_argument("--driver-id", default="car01")
    ap.add_argument("--out", type=Path, default=Path("traversals.csv"))
    args = ap.parse_args()
    trav = ingest(args.log_dir, args.driver_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trav.to_csv(args.out, index=False)
    print(f"{len(trav)} traversals, {int(trav['hard_brakes'].sum())} hard brakes -> {args.out}")


if __name__ == "__main__":
    main()
