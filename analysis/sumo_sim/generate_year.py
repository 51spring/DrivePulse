"""Generates a 1-year labelled commuting dataset with SUMO (weekday mornings and evenings).

    python sumo_sim/generate_year.py --weeks 52 --drivers 100 --seed 3 --out out/year/y_gangnam_s3
    python sumo_sim/generate_year.py --weeks 8 --weather-mix uniform --out out/year/pilot   # effect sizes

On top of each driver's base style (normal / drowsy / aggressive), life.py plans weather,
Korean holidays, usual departure times, daily form, vacations, route changes, new vehicles,
one-day anomalies and long-term drifts of three shapes (see life.py).

Every simulation window is saved as soon as it finishes (windows/), so an interrupted run
continues where it stopped when started again with the same arguments.

Outputs (in --out), same formats as generate.py so personal.py can read them:
    traversals.csv, trips.csv (+ weather, state_kind, strength, route_changed, vehicle_changed),
    events.csv, truth_drivers.csv, truth_day_anomalies.csv, truth_drift.csv (+ shape, ramp/hold/fall,
    end_date), truth_life.csv (vacation / route_change / vehicle_change), truth_calendar.csv,
    truth_incidents.csv
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from generate import BG_RATE, DRAIN, busy_edges  # noqa: E402
from life import (HOLIDAYS, WEATHER_TRAFFIC, apply_weather, depart_time, plan_lives, trip_style,  # noqa: E402
                  weather_calendar, workdays)
from run_window import run_window  # noqa: E402
from scenario import _style_params, _vtype_xml, build_network, fringe_edges, make_population  # noqa: E402
from baseline import add_context  # noqa: E402

WINDOWS = {"am": (7, 9), "pm": (17, 19)}
COMMUTE_P = 0.95      # on a working day (vacation / days off are planned separately)


def background_types(seed: int = 5) -> dict:
    """Same background vTypes as scenario.write_vtypes (before weather)."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in range(20):
        label = rng.choice(["normal", "drowsy", "aggressive"], p=[0.8, 0.1, 0.1])
        out[f"bg_{k:02d}"] = _style_params(label, rng)
    return out


def new_workplaces(net_file: Path, drivers: pd.DataFrame, ids: list[str], rng) -> dict:
    """driver -> (work, pm_from, pm_to) of a new workplace (job change)."""
    import sumolib
    net = sumolib.net.readNet(str(net_file))
    sources, sinks = fringe_edges(net_file)
    xy = lambda node: np.array(net.getNode(node).getCoord())
    nearest = lambda node, pool: min(pool, key=lambda e: np.linalg.norm(xy(pool[e]) - xy(node)))
    sink_ids = list(sinks)
    d = drivers.set_index("driver_id")
    out = {}
    for did in ids:
        home, old = d.loc[did, "home"], d.loc[did, "work"]
        while True:
            work = sink_ids[rng.integers(len(sink_ids))]
            if sinks[work] != sources[home] and sinks[work] != sinks[old]:
                break
        out[did] = (work, nearest(sinks[work], sources), nearest(sources[home], sinks))
    return out


class Plan:
    def __init__(self, args):
        self.args = args
        self.net = build_network(args.out / "net", "osm", args.osm_area)
        self.drivers = make_population(self.net, n_drivers=args.drivers, seed=args.pop_seed)
        rng = np.random.default_rng(args.seed)
        self.dates = pd.date_range(args.start, periods=args.weeks * 7, freq="D")
        self.workdays = workdays(self.dates)
        self.weather = weather_calendar(self.dates, rng, args.weather_mix)
        self.lives = plan_lives(list(self.drivers["driver_id"]), len(self.workdays), rng, WINDOWS,
                                drift_share=args.drift_share, route_share=args.route_share,
                                vehicle_share=args.vehicle_share, anomalies_per_year=args.anomalies_per_year)
        movers = [d for d, life in self.lives.items() if life.route_change is not None]
        self.new_work = new_workplaces(self.net, self.drivers, movers, rng)
        self.sources, self.sinks = fringe_edges(self.net)
        self.base = self.drivers.set_index("driver_id").to_dict("index")
        self.bg = background_types()
        candidates = busy_edges(self.net, self.drivers)
        self.incidents = {}
        for i, ds in enumerate(self.workdays):
            if rng.random() < args.incident_rate:
                w = str(rng.choice(list(WINDOWS)))
                b, e = WINDOWS[w]
                t0 = float(rng.uniform(b * 3600, e * 3600 - 1800))
                self.incidents[(ds, w)] = {"edge": str(rng.choice(candidates)), "t0": t0, "t1": t0 + 1200, "p": 0.6}
        (args.out / "vtypes").mkdir(parents=True, exist_ok=True)

    def window(self, i: int, w: str, write_vtypes: bool) -> tuple[dict, list[dict]]:
        """Simulation task for workday i x window w, and the planned named-driver trips."""
        ds = self.workdays[i]
        weather = self.weather[ds]
        rng = np.random.default_rng([self.args.seed, i, list(WINDOWS).index(w)])
        b, e = WINDOWS[w]
        t0, t1 = b * 3600, e * 3600 - 600
        trips, planned, types = [], [], {}
        for did, life in self.lives.items():
            if i in life.off_days or rng.random() > COMMUTE_P:
                continue
            d = self.base[did]
            moved = life.route_change is not None and i >= life.route_change
            if moved:
                work, pm_from, pm_to = self.new_work[did]
            else:
                work, pm_from, pm_to = d["work"], d["pm_from"], d["pm_to"]
            frm, to = (d["home"], work) if w == "am" else (pm_from, pm_to)
            params, kind, s = trip_style(d, life, i, w, weather, rng)
            vt = f"{did}@{ds}{w}"
            types[vt] = params
            vid = f"{did}#{ds}{w}"
            trips.append({"veh_id": vid, "driver_id": did, "vtype": vt, "from": frm, "to": to,
                          "trip_kind": "commute", "depart": depart_time(life, w, WINDOWS, rng)})
            planned.append({"trip_id": vid, "from": frm, "to": to, "trip_kind": "commute", "vtype": vt,
                            "state": kind is not None, "state_kind": kind or "", "strength": round(s, 3),
                            "weather": weather, "route_changed": moved,
                            "vehicle_changed": life.vehicle_change is not None and i >= life.vehicle_change})
        src, snk = list(self.sources), list(self.sinks)
        rate = BG_RATE[("weekday", w)] * WEATHER_TRAFFIC[weather]
        for k in range(rng.poisson(rate * (t1 - t0) / 3600)):
            frm, to = src[rng.integers(len(src))], snk[rng.integers(len(snk))]
            if self.sinks[to] == self.sources[frm]:
                continue
            trips.append({"veh_id": f"bg#{ds}{w}#{k}", "driver_id": "", "vtype": f"bg_{rng.integers(20):02d}",
                          "from": frm, "to": to, "depart": rng.uniform(t0, t1)})
        trips.sort(key=lambda x: x["depart"])
        vfile = self.args.out / "vtypes" / f"{ds}_{w}.add.xml"
        if write_vtypes:
            lines = ["<additional>"] + [_vtype_xml(k, p) for k, p in types.items()] + \
                    [_vtype_xml(k, apply_weather(p, weather)) for k, p in self.bg.items()] + ["</additional>"]
            vfile.write_text("\n".join(lines), encoding="utf-8")
        inc = self.incidents.get((ds, w))
        task = {"net": str(self.net), "vtypes": str(vfile), "date": ds, "window": w, "begin": b * 3600,
                "end": e * 3600 + DRAIN, "trips": trips, "incidents": [inc] if inc else [],
                "seed": int(rng.integers(1 << 30))}
        return task, planned

    def write_truth(self):
        out = self.args.out
        self.drivers.assign(**{f"depart_{w}": [self.lives[d].depart_mean[w] for d in self.drivers["driver_id"]]
                               for w in WINDOWS}).to_csv(out / "truth_drivers.csv", index=False)
        wd = self.workdays
        day, drift, life_rows = [], [], []
        for did, life in self.lives.items():
            for (i, w), kind in sorted(life.anomalies.items()):
                day.append({"driver_id": did, "date": wd[i], "window": w, "kind": kind, "strength": 1.0})
            if life.drift:
                dr = life.drift
                end = dr.end()
                drift.append({"driver_id": did, "kind": dr.kind, "shape": dr.shape, "start_date": wd[dr.start],
                              "ramp_days": dr.ramp, "hold_days": dr.hold, "fall_days": dr.fall,
                              "end_date": wd[end] if end is not None and end < len(wd) else ""})
            if life.vacation:
                life_rows.append({"driver_id": did, "event": "vacation", "start_date": wd[life.vacation[0]],
                                  "end_date": wd[life.vacation[1]]})
            if life.route_change is not None:
                life_rows.append({"driver_id": did, "event": "route_change", "start_date": wd[life.route_change],
                                  "end_date": ""})
            if life.vehicle_change is not None:
                life_rows.append({"driver_id": did, "event": "vehicle_change",
                                  "start_date": wd[life.vehicle_change], "end_date": ""})
        pd.DataFrame(day, columns=["driver_id", "date", "window", "kind", "strength"]) \
            .to_csv(out / "truth_day_anomalies.csv", index=False)
        pd.DataFrame(drift, columns=["driver_id", "kind", "shape", "start_date", "ramp_days", "hold_days",
                                     "fall_days", "end_date"]).to_csv(out / "truth_drift.csv", index=False)
        pd.DataFrame(life_rows, columns=["driver_id", "event", "start_date", "end_date"]) \
            .to_csv(out / "truth_life.csv", index=False)
        pd.DataFrame([{"date": d.strftime("%Y-%m-%d"), "weekday": d.dayofweek,
                       "holiday": HOLIDAYS.get(d.strftime("%Y-%m-%d"), ""),
                       "workday": d.strftime("%Y-%m-%d") in set(wd), "weather": self.weather[d.strftime("%Y-%m-%d")]}
                      for d in self.dates]).to_csv(out / "truth_calendar.csv", index=False)
        inc = pd.DataFrame([{"segment_id": v["edge"], "date": ds, "hour": int(v["t0"] // 3600),
                             "start_s": round(v["t0"]), "end_s": round(v["t1"])} for (ds, w), v in self.incidents.items()],
                           columns=["segment_id", "date", "hour", "start_s", "end_s"])
        if len(inc):
            inc["bucket"] = add_context(inc.assign(trip_id="", driver_id="", hard_brakes=0))["bucket"]
        inc.to_csv(out / "truth_incidents.csv", index=False)


def window_files(out: Path, ds: str, w: str) -> dict:
    base = out / "windows" / f"{ds}_{w}"
    return {k: base.with_name(f"{base.name}.{k}.csv") for k in ("traversals", "events", "trips")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2026-01-05")   # a Monday
    ap.add_argument("--weeks", type=int, default=52)
    ap.add_argument("--drivers", type=int, default=100)
    ap.add_argument("--osm-area", default="gangnam")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--pop-seed", type=int, default=11)
    ap.add_argument("--weather-mix", choices=["seoul", "uniform"], default="seoul",
                    help="seoul: synthetic Seoul-like seasons; uniform: a third each of clear/rain/snow (pilot)")
    ap.add_argument("--incident-rate", type=float, default=0.6, help="probability of one road incident per workday")
    ap.add_argument("--drift-share", type=float, default=0.3)
    ap.add_argument("--route-share", type=float, default=0.1)
    ap.add_argument("--vehicle-share", type=float, default=0.1)
    ap.add_argument("--anomalies-per-year", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--out", type=Path, default=HERE.parent / "out" / "year" / "y_gangnam_s3")
    ap.add_argument("--plan-only", action="store_true", help="write the truth files, don't simulate")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    key = {k: str(v) for k, v in vars(args).items() if k not in ("workers", "out", "plan_only")}
    cfg = args.out / "config.json"
    if cfg.exists() and json.loads(cfg.read_text(encoding="utf-8")) != key:
        sys.exit(f"{cfg} differs from these arguments: use the same arguments to resume, or a new --out")
    cfg.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")

    plan = Plan(args)
    plan.write_truth()
    slots = [(i, w) for i in range(len(plan.workdays)) for w in WINDOWS]
    weather_days = pd.Series([plan.weather[d] for d in plan.workdays]).value_counts().to_dict()
    print(f"{len(plan.workdays)} workdays ({weather_days}), {len(slots)} windows, {args.drivers} drivers, "
          f"{len(plan.incidents)} incidents, {args.workers} workers")
    if args.plan_only:
        return
    (args.out / "windows").mkdir(exist_ok=True)

    planned_rows, todo = [], []
    for i, w in slots:
        done = window_files(args.out, plan.workdays[i], w)["trips"].exists()
        _, planned = plan.window(i, w, write_vtypes=False)
        planned_rows += planned
        if not done:
            todo.append((i, w))
    print(f"to simulate: {len(todo)} of {len(slots)} windows", flush=True)

    def tasks():
        for i, w in todo:
            yield plan.window(i, w, write_vtypes=True)[0]

    started = time.time()
    if todo:
        with mp.Pool(min(args.workers, len(todo))) as pool:
            for n, r in enumerate(pool.imap_unordered(run_window, tasks()), 1):
                f = window_files(args.out, r["date"], "am" if r["begin"] // 3600 == WINDOWS["am"][0] else "pm")
                r["traversals"].to_csv(f["traversals"], index=False)
                r["events"].to_csv(f["events"], index=False)
                r["trips"].to_csv(f["trips"], index=False)       # last: marks the window done
                el = time.time() - started
                print(f"  [{n}/{len(todo)}] {r['date']} {r['begin'] // 3600:02d}h: "
                      f"{len(r['traversals']):,} traversals, {len(r['events']):,} hard brakes "
                      f"({el:.0f}s, ~{el / n * (len(todo) - n) / 60:.0f} min left)", flush=True)

    merge(args.out, plan, slots, pd.DataFrame(planned_rows))
    print(f"\n완료: {time.time() - started:.0f}s -> {args.out}")


def merge(out: Path, plan: Plan, slots, planned: pd.DataFrame):
    parts = {"traversals": [], "events": [], "trips": []}
    for i, w in slots:
        for k, f in window_files(out, plan.workdays[i], w).items():
            if f.stat().st_size > 2:
                parts[k].append(pd.read_csv(f, dtype={"segment_id": str, "driver_id": str}))
    trav = pd.concat(parts["traversals"], ignore_index=True)
    trav["driver_id"] = trav["driver_id"].fillna("").where(trav["driver_id"].fillna("") != "", "bg")
    trav.to_csv(out / "traversals.csv", index=False)
    pd.concat(parts["events"], ignore_index=True).to_csv(out / "events.csv", index=False)
    trips = pd.concat(parts["trips"], ignore_index=True).merge(planned, on="trip_id")
    trips.sort_values(["driver_id", "date", "window"]).to_csv(out / "trips.csv", index=False)
    summarize(trips, trav, plan)


def summarize(trips: pd.DataFrame, trav: pd.DataFrame, plan: Plan):
    """Effect sizes: weather, planted states, day-to-day spread of ordinary commutes."""
    t = trips.merge(plan.drivers[["driver_id", "label"]], on="driver_id")
    t = t[t["distance_m"].between(300, 1e6) & t["accel_rms"].between(0, 10)].copy()
    t["km"] = t["distance_m"] / 1000
    agg = lambda g: g.agg(trips=("trip_id", "size"), hard=("hard_brakes", "sum"), km=("km", "sum"),
                          moderate=("moderate_brakes", "sum"), accel_rms=("accel_rms", "mean"),
                          max_decel=("max_decel", "median")) \
        .assign(hard_per_km=lambda s: s.hard / s.km, moderate_per_km=lambda s: s.moderate / s.km) \
        .drop(columns=["hard", "km", "moderate"]).round(3)
    plain = t[~t["state"]]
    print("\n평소 출퇴근, 날씨별:")
    print(agg(plain.groupby(["weather"])).to_string())
    print("\n평소 출퇴근, 기본 성향 × 날씨:")
    print(agg(plain.groupby(["label", "weather"])).to_string())
    print("\n상태별 (맑은 날만):")
    clear = t[t["weather"] == "clear"].copy()
    clear["what"] = np.where(clear["state"], np.where(clear["strength"] >= 0.999, "full:", "partial:")
                             + clear["state_kind"], "평소")
    print(agg(clear.groupby(["label", "what"])).to_string())
    days = trav.merge(pd.DataFrame({"date": list(plan.weather), "weather": list(plan.weather.values())}), on="date")
    bg = days[days["driver_id"] == "bg"].groupby("weather")["hard_brakes"].agg(["size", "mean"])
    bg["per100"] = (100 * bg["mean"]).round(2)
    print("\n배경 교통 급정거 (100회 통과당), 날씨별:")
    print(bg[["size", "per100"]].to_string())
    # day-to-day spread of ordinary commutes (same driver, window, clear days, same route)
    p = plain[(plain["weather"] == "clear") & ~plain["route_changed"]]
    cv = p.groupby(["driver_id", "window"]).agg(n=("trip_id", "size"), hb_mean=("hard_brakes", "mean"),
                                                 hb_var=("hard_brakes", "var"), ar_mean=("accel_rms", "mean"),
                                                 ar_sd=("accel_rms", "std"), dep_sd=("depart_s", "std"))
    cv = cv[cv["n"] >= 5]
    print(f"\n평소 출퇴근 날마다의 변동 (맑은 날, 운전자×시간대 {len(cv)}개, 중앙값): "
          f"급정거 분산/평균 {np.nanmedian(cv['hb_var'] / cv['hb_mean'].clip(lower=0.05)):.2f}, "
          f"가속도 RMS 변동계수 {np.nanmedian(cv['ar_sd'] / cv['ar_mean']):.3f}, "
          f"출발 시각 표준편차 {np.nanmedian(cv['dep_sd']) / 60:.1f}분")


if __name__ == "__main__":
    main()
