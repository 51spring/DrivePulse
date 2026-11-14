"""Runs both detectors on a traversals CSV (from ingest_app.py or any other source).

Usage:
    python detect.py traversals.csv --out out/
Writes segment_anomalies.csv (flagged cells) and driver_scores.csv (all drivers).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baseline import add_context, detect_driver_anomalies, detect_segment_anomalies


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traversals", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--min-exposure", type=int, default=10, help="min traversals for a cell to be tested")
    ap.add_argument("--slot-minutes", type=int, default=None, help="e.g. 30 for 30-min slots instead of time bands")
    args = ap.parse_args()

    df = add_context(pd.read_csv(args.traversals), args.slot_minutes)
    cells = detect_segment_anomalies(df, min_exposure=args.min_exposure)
    drivers = detect_driver_anomalies(df, cells)

    args.out.mkdir(parents=True, exist_ok=True)
    cells[cells["flagged"]].to_csv(args.out / "segment_anomalies.csv", index=False)
    drivers.to_csv(args.out / "driver_scores.csv", index=False)

    tested = int((cells["exposure"] >= args.min_exposure).sum())
    print(f"traversals={len(df)}  cells={len(cells)}  tested cells={tested}")
    print(f"segment anomalies: {int(cells['flagged'].sum())}")
    print(cells[cells["flagged"]].head(10)[["segment_id", "bucket", "date", "exposure", "count", "expected", "rate_ratio", "q_value"]].to_string(index=False))
    print(f"\ndriver anomalies: {int(drivers['flagged'].sum())} / {len(drivers)}")
    print(drivers.head(10)[["driver_id", "traversals", "observed", "expected", "sir", "q_value", "flagged"]].to_string(index=False))
    if tested == 0:
        print("\n(no cell has enough traffic to test: the segment baseline needs many vehicles/days)")


if __name__ == "__main__":
    main()
