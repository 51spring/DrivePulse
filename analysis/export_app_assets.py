"""Exports what the car app needs from a SUMO/OSM network so that the app, the simulator
and the analysis all use the same road-segment IDs.

    python export_app_assets.py --area gangnam

Writes to app/src/main/assets/:
  segments_<area>.csv   edge_id, name, type, lanes, speed_kmh, "lon lat;lon lat;..." (map matching)
  demo_trip_<area>.csv  t_s, lat, lon, speed_mps, heading_deg, edge_id — one simulated car crossing
                        the area at 10 Hz, with an obstacle on its route so it brakes hard
                        (segment warning demo)

    python export_app_assets.py --area gangnam --personal-demo out/personal/p_gangnam_s3

  demo_history_<area>.csv  one normal driver's past morning commutes (personal-baseline demo)
  demo_commute_<area>.csv  the same driver's commute on a tired day, 10 Hz

    python export_app_assets.py --area gangnam --drift-demo out/personal/p_gangnam_s3
  demo_history_drift_<area>.csv  demo history + 12 commutes getting more fatigued (long-term demo)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sumo_sim"))
ASSETS = HERE.parent / "app" / "src" / "main" / "assets"


def export_segments(net_file: Path, out: Path) -> int:
    # sumolib needs pyproj for geo conversion; libsumo has PROJ built in.
    import libsumo as traci
    import sumolib
    net = sumolib.net.readNet(str(net_file))
    traci.start(["sumo", "-n", str(net_file), "--no-step-log", "true", "--no-warnings", "true"])
    n = 0
    try:
        _write_segments(net, traci, out)
        n = sum(1 for _ in out.open(encoding="utf-8")) - 1
    finally:
        traci.close()
    return n


def _write_segments(net, traci, out: Path):
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["edge_id", "name", "type", "lanes", "speed_kmh", "shape"])
        for e in net.getEdges():
            if e.getFunction() == "internal" or not e.allows("passenger"):
                continue
            shape = ";".join(f"{lon:.6f} {lat:.6f}" for lon, lat in
                             (traci.simulation.convertGeo(x, y) for x, y in e.getShape()))
            w.writerow([e.getID(), e.getName(), e.getType().replace("highway.", ""), e.getLaneNumber(),
                        round(e.getSpeed() * 3.6), shape])


def export_demo_trip(net_file: Path, out: Path, route_from: str, route_to: str, obstacle_edge: str | None,
                     ego_params: dict | None = None, seed: int = 1):
    """Simulates 25 min of morning traffic and records one car at 10 Hz.
    ego_params: driving style of the recorded car (default: a normal driver)."""
    import libsumo as traci
    import sumolib
    from scenario import fringe_edges, write_vtypes, make_population

    net = sumolib.net.readNet(str(net_file))
    work = HERE / "out" / "_demo_tmp"
    work.mkdir(exist_ok=True)
    drivers = make_population(net_file, n_drivers=10)
    vtypes = write_vtypes(drivers, work / "types.add.xml", variants={"ego_type": ego_params} if ego_params else None)
    sources, sinks = fringe_edges(net_file)
    rng = np.random.default_rng(seed)

    traci.start(["sumo", "-n", str(net_file), "-a", str(vtypes), "--begin", "25200", "--end", "27000",
                 "--step-length", "0.1", "--no-step-log", "true", "--no-warnings", "true",
                 "--time-to-teleport", "180", "--default.action-step-length", "0.1", "--seed", str(seed)])
    try:
        routes = {}

        def add(vid, frm, to, depart, vtype):
            key = (frm, to)
            if key not in routes:
                edges = traci.simulation.findRoute(frm, to).edges
                if not edges:
                    return False
                routes[key] = f"r{len(routes)}"
                traci.route.add(routes[key], edges)
            traci.vehicle.add(vid, routes[key], typeID=vtype, depart=f"{depart:.1f}", departLane="best",
                              departSpeed="max")
            return True

        src, snk = list(sources), list(sinks)
        for k in range(rng.poisson(1500 * 25 / 60)):  # background, 1500 veh/h
            add(f"bg{k}", src[rng.integers(len(src))], snk[rng.integers(len(snk))],
                rng.uniform(25200, 26700), f"bg_{rng.integers(20):02d}")
        normal = "ego_type" if ego_params else drivers[drivers["label"] == "normal"].iloc[0]["driver_id"]
        if not add("ego", route_from, route_to, 25800, normal):
            raise RuntimeError(f"no route {route_from} -> {route_to}")
        ego_route = traci.simulation.findRoute(route_from, route_to).edges

        rows, braked = [], False
        while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < 27000:
            traci.simulationStep()
            t = traci.simulation.getTime()
            if "ego" not in traci.vehicle.getIDList():
                if rows:
                    break
                continue
            road = traci.vehicle.getRoadID("ego")
            v = traci.vehicle.getSpeed("ego")
            # obstacle: when the ego car reaches the middle of the obstacle edge, it must stop hard
            if obstacle_edge and road == obstacle_edge and not braked and v > 6 and \
                    traci.vehicle.getLanePosition("ego") > 0.4 * traci.lane.getLength(f"{road}_0"):
                traci.vehicle.setSpeedMode("ego", 31 & ~4)
                traci.vehicle.slowDown("ego", 1.0, max((v - 1.0) / 5.5, 0.4))
                braked = True
            x, y = traci.vehicle.getPosition("ego")
            lon, lat = traci.simulation.convertGeo(x, y)
            rows.append((round(t - 25800, 1), f"{lat:.7f}", f"{lon:.7f}", round(v, 3),
                         round(traci.vehicle.getAngle("ego"), 1), road))
    finally:
        traci.close()

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "lat", "lon", "speed_mps", "heading_deg", "edge_id"])
        t0 = rows[0][0]
        for r in rows:
            w.writerow([round(r[0] - t0, 1), *r[1:]])
    return len(rows), ego_route


def main():
    from osm_korea import PRESETS, build_osm_network
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default="gangnam", choices=list(PRESETS))
    ap.add_argument("--from-edge", help="demo route start (default: west entry on Teheran-ro side)")
    ap.add_argument("--to-edge", help="demo route end")
    ap.add_argument("--obstacle-edge", help="edge where the demo car meets an obstacle")
    ap.add_argument("--personal-demo", type=Path,
                    help="SUMO dataset folder: export one normal driver's commute history and a "
                         "'tired day' commute on the same route (personal-baseline demo)")
    ap.add_argument("--drift-demo", type=Path,
                    help="same dataset as --personal-demo: add commutes that get more fatigued day by day "
                         "after the demo history (long-term change demo)")
    args = ap.parse_args()

    *bbox, _ = PRESETS[args.area]
    net_file = build_osm_network(tuple(bbox), HERE / "out" / "osm", args.area)
    ASSETS.mkdir(parents=True, exist_ok=True)
    n = export_segments(net_file, ASSETS / f"segments_{args.area}.csv")
    print(f"segments: {n} -> {ASSETS / f'segments_{args.area}.csv'}")

    if args.personal_demo:
        export_personal_demo(net_file, args.personal_demo, args.area)
        return
    if args.drift_demo:
        export_drift_history(net_file, args.drift_demo, args.area)
        return

    frm, to, obstacle = args.from_edge, args.to_edge, args.obstacle_edge
    if not (frm and to):
        frm, to, obstacle = pick_teheran_route(net_file)
    rows, route = export_demo_trip(net_file, ASSETS / f"demo_trip_{args.area}.csv", frm, to, obstacle)
    print(f"demo trip: {rows} samples, route {frm} -> {to} ({len(route)} edges), obstacle on {obstacle}")


HISTORY_HEADER = ["date", "window", "route", "km", "hard_brakes", "moderate_brakes", "accel_rms", "max_decel",
                  "e_usual", "e_today"]


def summarize_trace(csv_file: Path) -> dict:
    """The app's trip summary (DriveRecorder) of a 10 Hz demo trace: km, hard / moderate
    brakes, accel RMS while moving, max deceleration."""
    import pandas as pd
    from hard_brake import HardBrakeDetector
    d = pd.read_csv(csv_file)
    hard, moderate = HardBrakeDetector(), HardBrakeDetector(start_decel=2.0, end_decel=1.0)
    n_hard = n_mod = 0
    dist, sq, n, max_dec = 0.0, 0.0, 0, 0.0
    prev_t = prev_v = None
    for t, v in zip(d["t_s"], d["speed_mps"]):
        if prev_t is not None and t > prev_t:
            dt = t - prev_t
            dist += (prev_v + v) / 2 * dt
            if prev_v > 0.5 or v > 0.5:
                sq += ((v - prev_v) / dt) ** 2
                n += 1
        n_mod += moderate.on_sample(t, v) is not None
        n_hard += hard.on_sample(t, v) is not None
        if hard.last_accel is not None:
            max_dec = max(max_dec, -hard.last_accel)
        prev_t, prev_v = t, v
    return {"km": dist / 1000, "hard_brakes": n_hard, "moderate_brakes": n_mod,
            "accel_rms": (sq / n) ** 0.5 if n else 0.0, "max_decel": max_dec}


def export_drift_history(net_file: Path, dataset: Path, area: str, days: int = 12):
    """Long-term change demo: the personal demo driver's past commutes followed by `days`
    commutes on the same route that get more and more fatigued -> demo_history_drift_<area>.csv.
    Road expectations of the new days are the driver's usual ones (the road didn't change)."""
    import pandas as pd
    from scenario import style_variant
    hist = pd.read_csv(ASSETS / f"demo_history_{area}.csv")
    did, params, _ = pick_demo_driver(dataset)
    frm, to = hist["route"].iloc[0].split(">")
    print(f"drift demo: driver {did}, route {frm} -> {to}")
    rows = hist.to_dict("records")
    dates = pd.bdate_range(pd.Timestamp(hist["date"].max()) + pd.Timedelta(days=1), periods=days)
    tmp = HERE / "out" / "drift_demo_trip.csv"
    for k, date in enumerate(dates):
        strength = min(1.0, 0.5 + 0.5 * k / max(days // 2, 1))
        export_demo_trip(net_file, tmp, frm, to, None, ego_params=style_variant(params, "fatigue", strength),
                         seed=100 + k)
        s = summarize_trace(tmp)
        rows.append({"date": date.strftime("%Y-%m-%d"), "window": "am", "route": f"{frm}>{to}",
                     "km": round(s["km"], 3), "hard_brakes": s["hard_brakes"], "moderate_brakes": s["moderate_brakes"],
                     "accel_rms": round(s["accel_rms"], 4), "max_decel": round(s["max_decel"], 3),
                     "e_usual": round(hist["e_usual"].median(), 4), "e_today": round(hist["e_today"].median(), 4)})
        print(f"  {rows[-1]['date']} fatigue {strength:.2f}: {s['hard_brakes']} hard, {s['moderate_brakes']} moderate, "
              f"rms {s['accel_rms']:.2f}, max decel {s['max_decel']:.1f}")
    pd.DataFrame(rows)[HISTORY_HEADER].to_csv(ASSETS / f"demo_history_drift_{area}.csv", index=False)
    print(f"drift demo history: {len(hist)} usual + {days} fatigued commutes -> demo_history_drift_{area}.csv")


def pick_demo_driver(dataset: Path):
    """A normal driver (no planted anomalies or drift) with the longest morning commute:
    (driver_id, style parameters, their morning commutes)."""
    import pandas as pd
    from personal import load
    drivers = pd.read_csv(dataset / "truth_drivers.csv")
    skip = set(pd.read_csv(dataset / "truth_day_anomalies.csv")["driver_id"]) |         set(pd.read_csv(dataset / "truth_drift.csv")["driver_id"])
    t = load(dataset)
    am = t[(t["window"] == "am") & ~t["driver_id"].isin(skip)]
    am = am.merge(drivers[["driver_id", "label"]], on="driver_id")
    am = am[am["label"] == "normal"]
    stats = am.groupby("driver_id").agg(n=("trip_id", "size"), km=("km", "median"), hard=("hard_brakes", "mean"))
    stats = stats[(stats["n"] >= 15) & (stats["km"] >= 2.0)].sort_values("km", ascending=False)
    if stats.empty:
        raise RuntimeError("no suitable demo driver")
    did = stats.index[0]
    return did, drivers.set_index("driver_id").loc[did].to_dict(), am[am["driver_id"] == did].sort_values("date")


def export_personal_demo(net_file: Path, dataset: Path, area: str, min_brakes: int = 3):
    """Picks a normal driver (no planted anomalies or drift) with a long morning commute and
    writes (1) their past morning commutes as the app's history format and (2) a new morning
    commute on the same route driven 'fatigued' (scenario.style_variant)."""
    import pandas as pd
    from hard_brake import HardBrakeDetector
    from personal import load
    from scenario import style_variant

    did, params, hist = pick_demo_driver(dataset)
    frm, to = hist["from"].iloc[0], hist["to"].iloc[0]

    with (ASSETS / f"demo_history_{area}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "window", "route", "km", "hard_brakes", "moderate_brakes", "accel_rms", "max_decel",
                    "e_usual", "e_today"])
        for r in hist.itertuples():
            w.writerow([r.date, "am", f"{frm}>{to}", f"{r.km:.3f}", r.hard_brakes, r.moderate_brakes,
                        f"{r.accel_rms:.4f}", f"{r.max_decel:.3f}", f"{r.e_usual:.4f}", f"{r.e_today:.4f}"])
    print(f"demo driver {did}: {len(hist)} past morning commutes, {hist['km'].median():.1f} km, "
          f"mean {hist['hard_brakes'].mean():.2f} hard brakes -> demo_history_{area}.csv")

    tired = style_variant(params, "fatigue", 1.0)
    out = ASSETS / f"demo_commute_{area}.csv"   # demo_trip_<area>.csv stays the obstacle drive
    for seed in range(1, 11):   # a tired day that actually shows up in the braking
        rows, route = export_demo_trip(net_file, out, frm, to, None, ego_params=tired, seed=seed)
        d = pd.read_csv(out)
        det = HardBrakeDetector()
        n = sum(1 for ts, v in zip(d["t_s"], d["speed_mps"]) if det.on_sample(ts, v))
        print(f"  seed {seed}: tired-day commute {d['t_s'].max():.0f}s, {n} hard brakes")
        if n >= min_brakes:
            break
    print(f"demo trip ({did}, fatigued): {rows} samples, route {frm} -> {to} ({len(route)} edges)")


def pick_teheran_route(net_file: Path):
    """West-to-east trip whose route uses the most Teheran-ro edges; obstacle mid-route."""
    import sumolib
    from scenario import fringe_edges
    net = sumolib.net.readNet(str(net_file))
    sources, sinks = fringe_edges(net_file)
    best = None
    for s in sources:
        for k in sinks:
            path, _ = net.getShortestPath(net.getEdge(s), net.getEdge(k))
            if not path:
                continue
            teheran = [e for e in path if e.getName() == "테헤란로"]
            # obstacle on a long segment: very short edges are crossed within one 0.2 s
            # simulation step and never appear in traversal data
            long_ones = [e for e in teheran if e.getLength() >= 100]
            score = (len(teheran), -len(path))
            if long_ones and (best is None or score > best[0]):
                best = (score, s, k, long_ones[len(long_ones) // 2].getID())
    if best is None:
        raise RuntimeError("no route along 테헤란로 found")
    return best[1], best[2], best[3]


if __name__ == "__main__":
    main()
