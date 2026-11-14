"""Generates a labelled multi-day dataset with SUMO.

    python sumo_sim/generate.py --days 28 --workers 10 --out out/sumo
    python sumo_sim/generate.py --pilot          # one weekday morning, quick check

Drivers keep a persistent base style (normal / drowsy / aggressive) and commute every
weekday. On top of that, within-driver changes are planted:
  * one-day anomalies: a single commute (driver x date x am/pm) driven "fatigued" or
    "hurried" (scenario.style_variant)
  * long-term drift: from a start date a driver gradually becomes fatigued or hurried,
    reaching full strength after --drift-ramp-days and staying there

Outputs (in --out):
    traversals.csv           one row per vehicle x edge pass (input for baseline.py)
    trips.csv                one row per named-driver trip with driving features
    events.csv               every detected hard brake with speeds and deceleration
    truth_drivers.csv        driver_id, label (normal/drowsy/aggressive), style parameters
    truth_day_anomalies.csv  driver_id, date, window, kind, strength
    truth_drift.csv          driver_id, kind, start_date, ramp_days
    truth_incidents.csv      segment_id, date, bucket, start/end of planted road incidents
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from scenario import (STATE_KINDS, build_network, fringe_edges, make_population, road_classes,  # noqa: E402
                      style_variant, write_vtypes)
from run_window import run_window  # noqa: E402
from baseline import add_context  # noqa: E402

# name -> (begin hour, end hour). Departures stop 10 min before the end; +10 min to drain.
WINDOWS = {"am": (7, 9), "midday": (12, 13), "pm": (17, 19)}
# background vehicles per hour
BG_RATE = {
    ("weekday", "am"): 1700, ("weekday", "midday"): 700, ("weekday", "pm"): 1700,
    ("weekend", "am"): 400, ("weekend", "midday"): 1000, ("weekend", "pm"): 800,
}
COMMUTE_P = 0.9
WEEKEND_TRIP_P = 0.3
DRAIN = 600


def plan_states(args, drivers, dates, rng):
    """Within-driver changes. Returns ({(driver, date, window|None): (kind, strength)}, day_truth, drift_truth).
    window None = applies to every trip of that date (drift)."""
    states, day_rows, drift_rows = {}, [], []
    ids = list(drivers["driver_id"])
    weekdays = [d.strftime("%Y-%m-%d") for d in dates if d.dayofweek < 5]
    if args.pilot:  # one morning: a third fatigued, a third hurried -> effect size check
        for i, did in enumerate(ids):
            if i % 3:
                kind = STATE_KINDS[i % 3 - 1]
                states[(did, weekdays[0], "am")] = (kind, args.anomaly_strength)
                day_rows.append({"driver_id": did, "date": weekdays[0], "window": "am", "kind": kind,
                                 "strength": args.anomaly_strength})
        return states, day_rows, drift_rows

    drift_ids = set(rng.choice(ids, size=min(args.drift_drivers, len(ids)), replace=False))
    for did in sorted(drift_ids):
        kind = str(rng.choice(STATE_KINDS))
        start = int(rng.integers(args.drift_start_min, args.drift_start_max + 1))
        drift_rows.append({"driver_id": did, "kind": kind, "start_date": dates[start].strftime("%Y-%m-%d"),
                           "ramp_days": args.drift_ramp_days})
        for di in range(start, len(dates)):
            s = min(1.0, (di - start + 1) / args.drift_ramp_days)
            states[(did, dates[di].strftime("%Y-%m-%d"), None)] = (kind, s)

    # one-day anomalies after a week of history, on drivers without drift
    eligible = [(did, d, w) for did in ids if did not in drift_ids
                for d in weekdays[5:] for w in ("am", "pm")]
    for i in rng.choice(len(eligible), size=min(args.day_anomalies, len(eligible)), replace=False):
        did, d, w = eligible[i]
        kind = str(rng.choice(STATE_KINDS))
        states[(did, d, w)] = (kind, args.anomaly_strength)
        day_rows.append({"driver_id": did, "date": d, "window": w, "kind": kind, "strength": args.anomaly_strength})
    return states, day_rows, drift_rows


def state_vtype(states, driver_id, date, window):
    """vType id for this trip: a variant if the driver is in a changed state, else the base type."""
    if (driver_id, date, window) in states:
        return f"{driver_id}~{date}{window}"
    if (driver_id, date, None) in states:
        return f"{driver_id}~{date}"
    return driver_id


def variant_types(states, drivers) -> dict:
    base = drivers.set_index("driver_id").to_dict("index")
    out = {}
    for (did, date, window), (kind, s) in states.items():
        out[f"{did}~{date}{window or ''}"] = style_variant(base[did], kind, s)
    return out


def plan_trips(drivers, sources, sinks, date, weekend, window, rng, states=None):
    b, e = WINDOWS[window]
    t0, t1 = b * 3600, e * 3600 - 600
    src_ids, sink_ids = list(sources), list(sinks)
    trips = []

    for d in drivers.itertuples():
        frm = to = None
        kind = "commute"
        if not weekend and window == "am" and rng.random() < COMMUTE_P:
            frm, to = d.home, d.work
        elif not weekend and window == "pm" and rng.random() < COMMUTE_P:
            frm, to = d.pm_from, d.pm_to
        elif weekend and rng.random() < WEEKEND_TRIP_P:
            frm, to = src_ids[rng.integers(len(src_ids))], sink_ids[rng.integers(len(sink_ids))]
            kind = "other"
        if frm and sinks[to] != sources[frm]:
            trips.append({"veh_id": f"{d.driver_id}#{date}{window}", "driver_id": d.driver_id,
                          "vtype": state_vtype(states or {}, d.driver_id, date, window),
                          "from": frm, "to": to, "trip_kind": kind, "depart": rng.uniform(t0, t1)})

    n_bg = rng.poisson(BG_RATE[("weekend" if weekend else "weekday", window)] * (t1 - t0) / 3600)
    for k in range(n_bg):
        frm = src_ids[rng.integers(len(src_ids))]
        to = sink_ids[rng.integers(len(sink_ids))]
        if sinks[to] == sources[frm]:
            continue
        trips.append({"veh_id": f"bg#{date}{window}#{k}", "driver_id": "", "vtype": f"bg_{rng.integers(20):02d}",
                      "from": frm, "to": to, "depart": rng.uniform(t0, t1)})
    trips.sort(key=lambda x: x["depart"])
    return trips


def busy_edges(net_file, drivers, top=25):
    """Inner edges most used by commute routes (candidates for incidents)."""
    import sumolib
    net = sumolib.net.readNet(str(net_file))
    sources, sinks = fringe_edges(net_file)
    border = set(sources) | set(sinks)
    counts: dict[str, int] = {}
    for d in drivers.itertuples():
        path, _ = net.getShortestPath(net.getEdge(d.home), net.getEdge(d.work))
        for e in path or []:
            if e.getID() not in border:
                counts[e.getID()] = counts.get(e.getID(), 0) + 1
    return [e for e, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top]]


def build_tasks(args, net, vtypes, drivers, states, rng):
    sources, sinks = fringe_edges(net)
    candidates = busy_edges(net, drivers)
    dates = pd.date_range(args.start, periods=args.days, freq="D")
    tasks, incidents = [], []
    for di, date in enumerate(dates):
        weekend = date.dayofweek >= 5
        ds = date.strftime("%Y-%m-%d")
        windows = ["am"] if args.pilot else list(WINDOWS)
        inc_window = rng.choice(windows) if (not args.pilot and rng.random() < args.incident_rate) else None
        for w in windows:
            b, e = WINDOWS[w]
            inc = []
            if w == inc_window:
                t0 = rng.uniform(b * 3600, e * 3600 - 1800)
                inc = [{"edge": str(rng.choice(candidates)), "t0": t0, "t1": t0 + 1200, "p": 0.6}]
                incidents.append({"segment_id": inc[0]["edge"], "date": ds, "hour": int(t0 // 3600),
                                  "start_s": round(t0), "end_s": round(t0 + 1200)})
            tasks.append({"net": str(net), "vtypes": str(vtypes), "date": ds, "window": w,
                          "begin": b * 3600, "end": e * 3600 + DRAIN,
                          "trips": plan_trips(drivers, sources, sinks, ds, weekend, w, rng, states),
                          "incidents": inc, "seed": int(rng.integers(1 << 30))})
    return tasks, incidents


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=28)
    ap.add_argument("--start", default="2026-09-07")  # a Monday
    ap.add_argument("--drivers", type=int, default=300)
    ap.add_argument("--incident-rate", type=float, default=0.6, help="probability of one incident per day")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=Path, default=HERE.parent / "out" / "sumo")
    ap.add_argument("--pilot", action="store_true", help="one weekday morning only")
    ap.add_argument("--network", choices=["grid", "mixed", "osm"], default="grid",
                    help="grid: uniform signalised arterials; mixed: expressway + arterials + local road; "
                         "osm: real Korean roads from OpenStreetMap (--osm-area)")
    ap.add_argument("--osm-area", default="gangnam", help="preset in osm_korea.PRESETS (Korea only)")
    ap.add_argument("--pop-seed", type=int, default=11, help="seed of the driver population")
    ap.add_argument("--style-strength", type=float, default=1.0,
                    help="scale of drowsy/aggressive deviation from normal (1 = full, 0 = none)")
    ap.add_argument("--day-anomalies", type=int, default=60, help="one-day anomalous commutes to plant")
    ap.add_argument("--anomaly-strength", type=float, default=1.0, help="strength of one-day anomalies")
    ap.add_argument("--drift-drivers", type=int, default=20, help="drivers whose style drifts over weeks")
    ap.add_argument("--drift-start-min", type=int, default=10, help="earliest drift start (day index)")
    ap.add_argument("--drift-start-max", type=int, default=16, help="latest drift start (day index)")
    ap.add_argument("--drift-ramp-days", type=int, default=7, help="days until a drift reaches full strength")
    args = ap.parse_args()
    if args.pilot:
        args.days = 1

    args.out.mkdir(parents=True, exist_ok=True)
    net = build_network(args.out / "net", args.network, args.osm_area)
    drivers = make_population(net, n_drivers=args.drivers, seed=args.pop_seed,
                              style_strength=args.style_strength)
    rng = np.random.default_rng(args.seed)
    dates = pd.date_range(args.start, periods=args.days, freq="D")
    states, day_truth, drift_truth = plan_states(args, drivers, dates, rng)
    vtypes = write_vtypes(drivers, args.out / "net" / "drivers.add.xml", variants=variant_types(states, drivers))
    tasks, incidents = build_tasks(args, net, vtypes, drivers, states, rng)
    print(f"{len(tasks)} simulation windows, {sum(len(t['trips']) for t in tasks):,} trips, "
          f"{len(incidents)} incidents, {len(day_truth)} one-day anomalies, {len(drift_truth)} drifting drivers, "
          f"{args.workers} workers")

    started = time.time()
    with mp.Pool(min(args.workers, len(tasks))) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(run_window, tasks), 1):
            results.append(r)
            print(f"  [{i}/{len(tasks)}] {r['date']} {r['begin'] // 3600:02d}h: "
                  f"{len(r['traversals']):,} traversals, {len(r['events']):,} hard brakes "
                  f"({time.time() - started:.0f}s)", flush=True)

    trav = pd.concat([r["traversals"] for r in results], ignore_index=True)
    events = pd.concat([r["events"] for r in results], ignore_index=True)
    trav["driver_id"] = trav["driver_id"].where(trav["driver_id"] != "", "bg")
    trav.to_csv(args.out / "traversals.csv", index=False)
    events.to_csv(args.out / "events.csv", index=False)
    drivers.to_csv(args.out / "truth_drivers.csv", index=False)

    planned = pd.DataFrame([{k: t[k] for k in ("veh_id", "from", "to", "trip_kind", "vtype")}
                            for task in tasks for t in task["trips"] if t["driver_id"]]).rename(columns={"veh_id": "trip_id"})
    trips = pd.concat([r["trips"] for r in results if len(r["trips"])], ignore_index=True).merge(planned, on="trip_id")
    trips["state"] = trips["vtype"].str.contains("~")
    trips.sort_values(["driver_id", "date", "window"]).to_csv(args.out / "trips.csv", index=False)
    pd.DataFrame(day_truth, columns=["driver_id", "date", "window", "kind", "strength"]) \
        .to_csv(args.out / "truth_day_anomalies.csv", index=False)
    pd.DataFrame(drift_truth, columns=["driver_id", "kind", "start_date", "ramp_days"]) \
        .to_csv(args.out / "truth_drift.csv", index=False)
    summarize_states(trips, drivers, day_truth, drift_truth)
    inc = pd.DataFrame(incidents, columns=["segment_id", "date", "hour", "start_s", "end_s"])
    if len(inc):
        ctx = add_context(inc.assign(trip_id="", driver_id="", hard_brakes=0))
        inc["bucket"] = ctx["bucket"]
    inc.to_csv(args.out / "truth_incidents.csv", index=False)

    summarize(trav, drivers, road_classes(net))
    print(f"\n완료: {time.time() - started:.0f}s -> {args.out}")


def summarize_states(trips, drivers, day_truth, drift_truth):
    """Per-trip driving features by (base style, planted state) - effect size check."""
    t = trips[trips["trip_kind"] == "commute"].merge(drivers[["driver_id", "label"]], on="driver_id")
    day = {(r["driver_id"], r["date"], r["window"]): r["kind"] for r in day_truth}
    drift = {r["driver_id"]: r["kind"] for r in drift_truth}
    t["state_kind"] = [
        ("하루:" + day[(d, dt, w)]) if (d, dt, w) in day else (("장기:" + drift[d]) if s and d in drift else "평소")
        for d, dt, w, s in zip(t["driver_id"], t["date"], t["window"], t["state"])
    ]
    t["km"] = t["distance_m"] / 1000
    g = t.groupby(["label", "state_kind"]).agg(
        trips=("trip_id", "size"), hard_per_km=("hard_brakes", "sum"), km=("km", "sum"),
        moderate=("moderate_brakes", "sum"), accel_rms=("accel_rms", "mean"), speed_cv=("speed_cv", "mean"),
        max_decel=("max_decel", "mean"))
    g["hard_per_km"] = g["hard_per_km"] / g["km"]
    g["moderate_per_km"] = g.pop("moderate") / g.pop("km")
    print("\n출퇴근 운행 특성 (기본 성향 × 상태):")
    print(g.round(3).to_string())


def summarize(trav, drivers, classes=None):
    lab = trav.merge(drivers[["driver_id", "label"]], on="driver_id", how="left")
    lab["label"] = lab["label"].fillna("background")
    rate = lambda g: g.agg(traversals=("hard_brakes", "size"), hard_brakes=("hard_brakes", "sum")) \
        .assign(per100=lambda s: 100 * s["hard_brakes"] / s["traversals"]).round(2)
    print("\n급정거율 (100회 통과당), 운전 성향별:")
    print(rate(lab.groupby("label")).to_string())
    if classes:
        lab["road_class"] = lab["segment_id"].map(classes)
        print("\n급정거율 (100회 통과당), 도로 종류별:")
        print(rate(lab.groupby("road_class")).to_string())


if __name__ == "__main__":
    main()
