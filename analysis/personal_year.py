"""One year of commutes: does the personal baseline stay quiet when life changes, and still
catch the days and the drifts it should? (datasets from sumo_sim/generate_year.py)

Thresholds are set ONCE on the first dataset (target false alarms per driver-year) and then
applied unchanged to the others, the way the app would ship them.

One-day anomalies (per commute)
  * false alarms per driver-year and recall, by method
  * rain / snow days: false alarms with and without correction
        personal_road     today's road from background traffic on the same segments
        personal_fleet_f  + how every app user drove this window today (trimmed)
  * history length: recall by how many past commutes the driver has, for three baselines
        expanding   all past commutes          window60  the last 60
        fixed20     the first 20 only (never updated)
  * new workplace / new car: false alarms after the change, history kept per window vs
    restarted per route

Long-term drift (per driver-day, one-sided CUSUM restarted after each alarm, daily z capped
so a single bad day can't raise it)
  * reference  frozen: first 20 commuting days      lagged: days i-80 .. i-20
  * detection and delay by drift shape (fast / slow / transient), recovery after a transient
  * alarms after a new workplace / new car

    python personal_year.py                 # all out/year/y_* -> docs/results/personal_year.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baseline import _BAND_OF_HOUR, cell_rates  # noqa: E402
from personal import (A_PERSONAL, FEATURES, METHOD_KO, MIN_HISTORY, N0_FEATURE, Z_CAP,  # noqa: E402
                      _past_sum, expectations_from, score_trips)

OUT = HERE / "out" / "year"
DOCS = HERE.parent / "docs"
AREA_KO = {"gangnam": "강남", "pangyo": "판교"}
YEAR_DAYS = 245.0                       # workdays in a year

DAY_METHODS = ["absolute", "personal_road", "personal_road_f", "personal_fleet_f"]
DAY_TARGET_FA = 2.0                     # false alarms per driver-year (one-day detector)
HISTORIES = {"expanding": None, "window60": ("window", 60), "fixed20": ("fixed", 20)}
POS_BINS = [10, 30, 60, 120, 250]       # one-day anomalies are planted after the first 10 workdays

DRIFT_METHODS = ["absolute", "personal_road_f", "personal_fleet_f"]
DRIFT_TARGET_FA = 0.5                   # drift alarms per driver-year
REF_DAYS, LAG_GAP, LAG_LEN = 20, 20, 60
REFERENCES = {"frozen": None, "lagged": (LAG_GAP, LAG_LEN)}
DAY_Z_CAP = 3.0
CUSUM_K = 0.5
AFTER_CHANGE = 60                       # workdays after a new workplace / car that are watched


# ---------------------------------------------------------------- loading

def year_expectations(folder: Path, chunk: int = 4_000_000) -> pd.DataFrame:
    """personal.trip_expectations, but counting the (multi-GB) traversals file chunk by chunk."""
    cache = folder / "trip_expectations.csv"
    if cache.exists():
        return pd.read_csv(cache)
    cols = ["trip_id", "driver_id", "segment_id", "date", "hour", "hard_brakes"]
    cells, own = [], []
    for ch in pd.read_csv(folder / "traversals.csv", usecols=cols, chunksize=chunk,
                          dtype={"segment_id": str, "date": str, "driver_id": str}):
        weekday = pd.to_datetime(pd.Series(ch["date"].unique()))
        day_type = dict(zip(ch["date"].unique(), np.where(weekday.dt.dayofweek >= 5, "weekend", "weekday")))
        ch["bucket"] = ch["date"].map(day_type) + "_" + _BAND_OF_HOUR[ch["hour"].astype(int).to_numpy()]
        cells.append(ch.groupby(["segment_id", "bucket", "date"])["hard_brakes"].agg(["size", "sum"]))
        own.append(ch[ch["driver_id"] != "bg"][["trip_id", "hard_brakes", "segment_id", "bucket", "date"]])
        print(f"  traversals read: {sum(len(o) for o in own)} own rows", flush=True)
    c = pd.concat(cells).groupby(level=[0, 1, 2]).sum().reset_index()
    c = c.rename(columns={"size": "exposure", "sum": "count"})
    c["parent_bucket"] = c["bucket"]
    exp = expectations_from(pd.concat(own), cell_rates(c))
    exp.to_csv(cache, index=False)
    return exp


def load_year(folder: Path) -> dict:
    trips = pd.read_csv(folder / "trips.csv")
    trips = trips[(trips["trip_kind"] == "commute") & (trips["distance_m"] > 300)].copy()
    ok = (trips["distance_m"] < 1e6) & (trips["accel_rms"].between(0, 10)) & (trips["max_decel"] < 1e3)
    trips = trips[ok].copy()
    trips["max_decel"] = trips["max_decel"].clip(upper=10.0)
    t = trips.merge(year_expectations(folder), on="trip_id", how="inner")
    t["km"] = t["distance_m"] / 1000
    t["moderate_per_km"] = t["moderate_brakes"] / t["km"]
    t["hard_per_km"] = t["hard_brakes"] / t["km"]
    t["route"] = t["from"].astype(str) + ">" + t["to"].astype(str)
    drivers = pd.read_csv(folder / "truth_drivers.csv")
    t = t.merge(drivers[["driver_id", "label"]], on="driver_id")
    t = t.sort_values(["driver_id", "window", "date"]).reset_index(drop=True)

    life = pd.read_csv(folder / "truth_life.csv")
    drift = pd.read_csv(folder / "truth_drift.csv")
    day = pd.read_csv(folder / "truth_day_anomalies.csv")
    cal = pd.read_csv(folder / "truth_calendar.csv")
    change = {e: dict(zip(g["driver_id"], g["start_date"])) for e, g in life.groupby("event")}
    route_ch, veh_ch = change.get("route_change", {}), change.get("vehicle_change", {})
    special = set(drift["driver_id"]) | set(route_ch) | set(veh_ch)
    keys = set(zip(day["driver_id"], day["date"], day["window"]))
    t["anomaly"] = [k in keys for k in zip(t["driver_id"], t["date"], t["window"])]
    kind_of = {(r.driver_id, r.date, r.window): r.kind for r in day.itertuples()}
    t["kind"] = [kind_of.get(k, "") for k in zip(t["driver_id"], t["date"], t["window"])]
    return {"trips": t, "drift": drift, "route_change": route_ch, "vehicle_change": veh_ch,
            "plain": sorted(set(drivers["driver_id"]) - special), "drivers": drivers,
            "n_workdays": int(cal["workday"].sum()), "calendar": cal}


# ---------------------------------------------------------------- one-day anomalies

def day_threshold(scored: pd.DataFrame, plain: list, n_workdays: int, method: str, target: float) -> float:
    neg = eligible(scored)
    neg = neg[neg["driver_id"].isin(plain) & ~neg["anomaly"]][f"s_{method}"].sort_values(ascending=False)
    driver_years = len(plain) * n_workdays / YEAR_DAYS
    k = int(round(target * driver_years))
    return float(neg.iloc[k]) if k < len(neg) else float("-inf")


def eligible(scored: pd.DataFrame) -> pd.DataFrame:
    return scored[scored["n_prev"] >= MIN_HISTORY]


def day_metrics(scored: pd.DataFrame, data: dict, thr: dict) -> dict:
    """False alarms (plain drivers) and recall (non-drift drivers) at fixed thresholds."""
    e = eligible(scored)
    drift_ids = set(data["drift"]["driver_id"])
    plain = e[e["driver_id"].isin(data["plain"]) & ~e["anomaly"]]
    pos = e[e["anomaly"] & ~e["driver_id"].isin(drift_ids)]
    dy = len(data["plain"]) * data["n_workdays"] / YEAR_DAYS
    res = {}
    for m in DAY_METHODS:
        s = f"s_{m}"
        res[f"fa_{m}"] = float((plain[s] > thr[m]).sum() / dy)
        res[f"recall_{m}"] = float((pos[s] > thr[m]).mean())
        for lab in ("normal", "drowsy", "aggressive"):
            g = pos[pos["label"] == lab]
            res[f"recall_{m}_{lab}"] = float((g[s] > thr[m]).mean()) if len(g) else float("nan")
        for kind in ("fatigue", "hurried"):
            g = pos[pos["kind"] == kind]
            res[f"recall_{m}_{kind}"] = float((g[s] > thr[m]).mean()) if len(g) else float("nan")
        for w in ("clear", "rain", "snow"):
            g = plain[plain["weather"] == w]
            res[f"fa100_{m}_{w}"] = float(100 * (g[s] > thr[m]).mean()) if len(g) else float("nan")
    res["n_pos"] = len(pos)
    return res


def by_position(scored: pd.DataFrame, data: dict, thr: float, method: str) -> pd.DataFrame:
    """Recall and false alarms per 100 commutes by how many commutes the driver has made."""
    e = eligible(scored).copy()
    e["bin"] = pd.cut(e["pos"], POS_BINS, right=False)
    drift_ids = set(data["drift"]["driver_id"])
    s = f"s_{method}"
    pos = e[e["anomaly"] & ~e["driver_id"].isin(drift_ids)].groupby("bin", observed=False)[s]
    neg = e[e["driver_id"].isin(data["plain"]) & ~e["anomaly"]].groupby("bin", observed=False)[s]
    return pd.DataFrame({"recall": pos.apply(lambda x: (x > thr).mean()),
                         "fa100": neg.apply(lambda x: 100 * (x > thr).mean())})


def change_alarms(scored: pd.DataFrame, changes: dict, thr: float, method: str) -> dict:
    """False alarms per 100 commutes in the first 30 commutes after a change vs before it."""
    s = f"s_{method}"
    before, after = [], []
    for did, date in changes.items():
        g = scored[(scored["driver_id"] == did) & ~scored["anomaly"]]
        b = g[g["date"] < date]
        b = b[b["n_prev"] >= MIN_HISTORY]
        a = g[g["date"] >= date].groupby("window").head(30)
        before.append((b[s] > thr).to_numpy())
        # a commute without enough history on its key is not scored -> no alarm
        after.append(((a[s] > thr) & (a["n_prev"] >= MIN_HISTORY)).to_numpy())
    cat = lambda x: np.concatenate(x) if x else np.array([])
    b, a = cat(before), cat(after)
    return {"before": 100 * b.mean() if len(b) else float("nan"), "after": 100 * a.mean() if len(a) else float("nan")}


# ---------------------------------------------------------------- long-term drift

def daily_year(t: pd.DataFrame) -> pd.DataFrame:
    agg = {"hard_brakes": "sum", "e_usual": "sum", "e_today": "sum", "km": "sum",
           "moderate_brakes": "sum", "accel_rms": "mean", "max_decel": "mean", "weather": "first",
           "strength": "max"}
    d = t.groupby(["driver_id", "label", "date"]).agg(agg).reset_index().sort_values(["driver_id", "date"])
    d["moderate_per_km"] = d["moderate_brakes"] / d["km"]
    d["hard_per_km"] = d["hard_brakes"] / d["km"]
    d["day_idx"] = d.groupby("driver_id").cumcount()
    return d.reset_index(drop=True)


def _ref_sum(x: np.ndarray, lag) -> np.ndarray:
    """Per day i: sum of x over the reference days (frozen: first REF_DAYS; lagged: [i-gap-len, i-gap),
    never earlier than the frozen window) and the number of those days."""
    n = len(x)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(n)
    if lag is None:
        b = np.full(n, min(REF_DAYS, n))
        a = np.zeros(n, dtype=int)
    else:
        gap, length = lag
        b = np.minimum(np.maximum(REF_DAYS, i - gap), n)
        a = np.maximum(0, b - length)
    return cs[b] - cs[a], (b - a).astype(float)


def drift_z(d: pd.DataFrame, reference) -> pd.DataFrame:
    """Daily z of every drift method vs each driver's reference days."""
    d = d.copy()
    parts = []
    for _, g in d.groupby("driver_id", sort=False):
        out = pd.DataFrame(index=g.index)
        y, _ = _ref_sum(g["hard_brakes"].to_numpy(float), reference)
        et, _ = _ref_sum(g["e_today"].to_numpy(float), reference)
        out["f"] = (y + A_PERSONAL) / (et + A_PERSONAL)
        for c in FEATURES:
            x = g[c].to_numpy(float)
            s1, n = _ref_sum(x, reference)
            s2, _ = _ref_sum(x**2, reference)
            out[f"m_{c}"] = s1 / n
            out[f"v_{c}"] = np.clip(s2 / n - (s1 / n) ** 2, 0, None) * n / np.maximum(n - 1, 1)
            out[f"n_{c}"] = n
        parts.append(out)
    d = d.join(pd.concat(parts))
    exp = d["f"] * d["e_today"]
    z_road = (d["hard_brakes"] - exp) / np.sqrt(exp + 0.25)
    zf = []
    for c in FEATURES:
        pop_var = d.loc[d["day_idx"] < REF_DAYS].groupby("driver_id")[c].var().median()
        v = (d[f"n_{c}"] * d[f"v_{c}"] + N0_FEATURE * pop_var) / (d[f"n_{c}"] + N0_FEATURE)
        zf.append(((d[c] - d[f"m_{c}"]) / np.sqrt(v.clip(lower=1e-9))).clip(-Z_CAP, Z_CAP))
    road_f = (z_road + sum(zf)) / np.sqrt(1 + len(FEATURES))
    d["z_personal_road_f"] = road_f.clip(-DAY_Z_CAP, DAY_Z_CAP)
    # fleet: how all users deviated from their own reference that day -> centre on the median
    # and scale by the spread across users (MAD), so the threshold carries over between areas
    by_day = road_f.groupby(d["date"])
    med = by_day.transform("median")
    mad = (road_f - med).abs().groupby(d["date"]).transform("median") * 1.4826
    fleet = (road_f - med) / mad.clip(lower=0.3)
    d["z_personal_fleet_f"] = fleet.clip(-DAY_Z_CAP, DAY_Z_CAP)
    return d


def alarms(d: pd.DataFrame, method: str, h: float) -> dict:
    """driver -> list of alarm day_idx (CUSUM restarted after each alarm)."""
    out = {}
    for did, g in d.groupby("driver_id", sort=False):
        g = g[g["day_idx"] >= REF_DAYS]
        days = g["day_idx"].to_numpy()
        hits = []
        if method == "absolute":   # rolling 5-day hard brakes per km above a fixed level, re-armed after 5 days
            stat = g["hard_per_km"].rolling(5, min_periods=3).mean().fillna(0).to_numpy()
            last = -99
            for k, v in enumerate(stat):
                if v > h and k - last >= 5:
                    hits.append(int(days[k]))
                    last = k
        else:
            s = 0.0
            for k, z in enumerate(g[f"z_{method}"].to_numpy()):
                s = max(0.0, s + z - CUSUM_K)
                if s > h:
                    hits.append(int(days[k]))
                    s = 0.0
        out[did] = hits
    return out


def fa_rate(d: pd.DataFrame, method: str, h: float, plain: list) -> float:
    a = alarms(d[d["driver_id"].isin(plain)], method, h)
    days = d[d["driver_id"].isin(plain) & (d["day_idx"] >= REF_DAYS)].shape[0]
    return sum(len(v) for v in a.values()) / max(days, 1) * YEAR_DAYS


def calibrate_drift(d: pd.DataFrame, method: str, plain: list, target: float) -> float:
    lo, hi = (0.0, 5.0) if method == "absolute" else (0.5, 60.0)
    for _ in range(30):
        mid = (lo + hi) / 2
        if fa_rate(d, method, mid, plain) > target:
            lo = mid
        else:
            hi = mid
    return hi


def date_to_idx(d: pd.DataFrame, did: str, date: str):
    g = d[d["driver_id"] == did]
    later = g[g["date"] >= date]
    return int(later["day_idx"].iloc[0]) if len(later) else None


def drift_metrics(d: pd.DataFrame, data: dict, h: dict) -> dict:
    res = {}
    last = d.groupby("driver_id")["day_idx"].max()
    for m in DRIFT_METHODS:
        a = alarms(d, m, h[m])
        res[f"fa_{m}"] = fa_rate(d, m, h[m], data["plain"])
        for shape, rows in data["drift"].groupby("shape"):
            det, delays, early, cleared, n_tr = 0, [], 0, 0, 0
            for r in rows.itertuples():
                s = date_to_idx(d, r.driver_id, r.start_date)
                if s is None:
                    continue
                e = date_to_idx(d, r.driver_id, r.end_date) if isinstance(r.end_date, str) else None
                end = e if e is not None else last[r.driver_id]
                hits = a[r.driver_id]
                early += any(x < s for x in hits)
                inside = [x for x in hits if s <= x <= end]
                if inside:
                    det += 1
                    delays.append(inside[0] - s)
                if shape == "transient" and e is not None:
                    n_tr += 1
                    cleared += not any(x >= e + 10 for x in hits)
            n = len(rows)
            res[f"detect_{m}_{shape}"] = det / n
            res[f"delay_{m}_{shape}"] = float(np.median(delays)) if delays else float("nan")
            res[f"early_{m}_{shape}"] = early / n
            if shape == "transient":
                res[f"cleared_{m}"] = cleared / max(n_tr, 1)
        for ev in ("route_change", "vehicle_change"):
            hit = 0
            for did, date in data[ev].items():
                s = date_to_idx(d, did, date)
                hit += s is not None and any(s <= x < s + AFTER_CHANGE for x in a[did])
            res[f"{ev}_{m}"] = hit / max(len(data[ev]), 1)
    # chance that a plain driver alarms within AFTER_CHANGE days anyway (for comparison)
    res["chance_after"] = {m: 1 - np.exp(-res[f"fa_{m}"] * AFTER_CHANGE / YEAR_DAYS) for m in DRIFT_METHODS}
    return res


# ---------------------------------------------------------------- the app's settings

# Mirror of app PersonalBaseline.kt: one-day check on the same route and window over the last
# APP_HISTORY commutes, hard brakes tested with the driver's own day-to-day spread; drift per
# (window, route) with the lagged reference and road correction, leaving out days whose road was
# unusually gentle (server's today/usual expectation < APP_ROAD_SKIP: snow - everyone slow), and
# counting notifications (an alarm after APP_RECENT quiet commute days). The fleet correction is
# left out: it would need each car's deviation on the server, which the app avoids.
APP_HISTORY = 60
APP_P = {"watch": 1e-3, "alert": 1e-4}
# PersonalBaseline.kt POP_VAR_*: median within-driver variances of the features on ordinary
# 1-year SUMO commutes (route x window groups); used while a driver has few past commutes
APP_POP_VAR = {"moderate_per_km": 0.10, "accel_rms": 0.014, "max_decel": 0.36}
APP_ROAD_SKIP = 0.7
APP_RECENT = 10
APP_DRIFT_H = 10.0   # all five datasets pooled (10.03): 0.5 notifications per unchanged driver-year (calibrate_app_drift)


def app_units(t: pd.DataFrame) -> pd.DataFrame:
    """Drift series per driver x window x route (one commute per day): driver_id holds the
    unit so drift_z / alarms work unchanged, 'who' the driver. Snow-like road days left out."""
    t = t[t["e_today"] / t["e_usual"] >= APP_ROAD_SKIP]
    d = t.groupby(["driver_id", "label", "window", "route", "date"]).agg(
        hard_brakes=("hard_brakes", "sum"), e_usual=("e_usual", "sum"), e_today=("e_today", "sum"),
        km=("km", "sum"), moderate_brakes=("moderate_brakes", "sum"), accel_rms=("accel_rms", "mean"),
        max_decel=("max_decel", "mean"), weather=("weather", "first"), strength=("strength", "max")).reset_index()
    d["moderate_per_km"] = d["moderate_brakes"] / d["km"]
    d["hard_per_km"] = d["hard_brakes"] / d["km"]
    d["who"] = d["driver_id"]
    d["driver_id"] = d["who"] + "|" + d["window"] + "|" + d["route"]
    d = d.sort_values(["driver_id", "date"]).reset_index(drop=True)
    d["day_idx"] = d.groupby("driver_id").cumcount()
    return d


def app_notifications(z: pd.DataFrame, h: float) -> dict:
    """unit -> dates the app would newly report a change (alarm after APP_RECENT quiet days)."""
    out = {}
    for unit, hits in alarms(z, "personal_road_f", h).items():
        dates = z[z["driver_id"] == unit].set_index("day_idx")["date"]
        kept, last = [], -10_000
        for x in hits:
            if x - last > APP_RECENT:
                kept.append(dates[x])
            last = x
        out[unit] = kept
    return out


def app_drift(z: pd.DataFrame, data: dict, h: float) -> dict:
    nt = app_notifications(z, h)
    plain = set(data["plain"])
    res = {"drift_fa": sum(len(v) for u, v in nt.items() if u.split("|")[0] in plain)
           / (len(plain) * data["n_workdays"] / YEAR_DAYS)}
    by_driver = {}
    for unit, dates in nt.items():
        by_driver.setdefault(unit.split("|")[0], []).extend(dates)
    for shape, rows in data["drift"].groupby("shape"):
        det, delays = 0, []
        for r in rows.itertuples():
            end = r.end_date if isinstance(r.end_date, str) else "9999"
            inside = sorted(x for x in by_driver.get(r.driver_id, []) if r.start_date <= x <= end)
            if inside:
                det += 1
                delays.append(np.busday_count(r.start_date, inside[0]))
        res[f"drift_detect_{shape}"] = det / len(rows)
        res[f"drift_delay_{shape}"] = float(np.median(delays)) if delays else float("nan")
    return res


def calibrate_app_drift(datasets: list[dict], target: float = DRIFT_TARGET_FA) -> float:
    """H giving `target` notifications per unchanged driver-year, averaged over the datasets
    (one dataset has only ~60 unchanged drivers: too few alarms to set H on its own)."""
    zs = [(drift_z(app_units(d["trips"]), REFERENCES["lagged"]), d) for d in datasets]
    lo, hi = 1.0, 30.0
    for _ in range(25):
        mid = (lo + hi) / 2
        fa = np.mean([app_drift(z, d, mid)["drift_fa"] for z, d in zs])
        lo, hi = (mid, hi) if fa > target else (lo, mid)
    return hi


def app_check(data: dict) -> dict:
    t = data["trips"]
    s = score_trips(t, keys=("driver_id", "window", "route"), history=("window", APP_HISTORY), dispersion=True,
                    pop_var=APP_POP_VAR)
    e = eligible(s)
    drift_ids = set(data["drift"]["driver_id"])
    plain = e[e["driver_id"].isin(data["plain"]) & ~e["anomaly"]]
    pos = e[e["anomaly"] & ~e["driver_id"].isin(drift_ids)]
    dy = len(data["plain"]) * data["n_workdays"] / YEAR_DAYS
    res = {}
    for name, p in APP_P.items():
        thr = -np.log(p)
        res[f"day_fa_{name}"] = float((plain["s_personal_road_f"] > thr).sum() / dy)
        res[f"day_recall_{name}"] = float((pos["s_personal_road_f"] > thr).mean())
        for lab in ("normal", "drowsy", "aggressive"):
            g = pos[pos["label"] == lab]
            res[f"day_recall_{name}_{lab}"] = float((g["s_personal_road_f"] > thr).mean()) if len(g) else float("nan")
    res.update(app_cause(s, drift_ids))
    res.update(app_drift(drift_z(app_units(t), REFERENCES["lagged"]), data, APP_DRIFT_H))
    return res


def app_cause(s: pd.DataFrame, drift_ids: set) -> dict:
    """The app's tired / rushed wording on alerted planted days: rushed when accel RMS moved
    further from own history (z, variance shrunk to APP_POP_VAR) than moderate braking."""
    g = s.groupby(["driver_id", "window", "route"], sort=False)
    n = s["n_prev"].clip(lower=1)
    z = {}
    for c in ("accel_rms", "moderate_per_km"):
        s = s.assign(_sq=s[c] ** 2)
        g = s.groupby(["driver_id", "window", "route"], sort=False)
        m = _past_sum(g, c, ("window", APP_HISTORY)) / n
        v = (_past_sum(g, "_sq", ("window", APP_HISTORY)) / n - m**2).clip(lower=0)
        v = (s["n_prev"] * v + N0_FEATURE * APP_POP_VAR[c]) / (s["n_prev"] + N0_FEATURE)
        z[c] = (s[c] - m) / np.sqrt(v)
    hit = (s["n_prev"] >= MIN_HISTORY) & s["anomaly"] & ~s["driver_id"].isin(drift_ids)         & (s["s_personal_road_f"] > -np.log(APP_P["alert"]))
    pred = np.where(z["accel_rms"] > z["moderate_per_km"], "hurried", "fatigue")
    ok = pd.Series(pred == s["kind"].to_numpy(), index=s.index)[hit]
    kinds = s.loc[hit, "kind"]
    return {"cause_acc": float(ok.mean()), "cause_acc_fatigue": float(ok[kinds == "fatigue"].mean()),
            "cause_acc_hurried": float(ok[kinds == "hurried"].mean()), "cause_n": int(hit.sum())}


# ---------------------------------------------------------------- run one dataset

def evaluate(folder: Path, cal: dict | None) -> tuple[dict, dict, dict]:
    """Returns (metrics, calibration, extras for plots). cal None -> calibrate on this dataset."""
    data = load_year(folder)
    t = data["trips"]
    first = cal is None
    cal = cal or {"day": {}, "drift": {}}
    res = {"run": folder.name, "n_trips": len(t), "n_drivers": t["driver_id"].nunique()}

    scored = {name: score_trips(t, history=hist) for name, hist in HISTORIES.items()}
    main = scored["expanding"]
    if first:
        cal["day"] = {m: day_threshold(main, data["plain"], data["n_workdays"], m, DAY_TARGET_FA)
                      for m in DAY_METHODS}
    res["day"] = {name: day_metrics(s, data, cal["day"]) for name, s in scored.items()}
    best = "personal_fleet_f"
    res["position"] = {name: by_position(s, data, cal["day"][best], best).reset_index().astype(
        {"bin": str}).to_dict("list") for name, s in scored.items()}
    by_route = score_trips(t, keys=("driver_id", "window", "route"))
    res["changes"] = {}
    for ev in ("route_change", "vehicle_change"):
        res["changes"][ev] = {
            "window": change_alarms(main, data[ev], cal["day"][best], best),
            "route": change_alarms(by_route, data[ev], cal["day"][best], best)}

    daily = daily_year(t)
    zs = {ref: drift_z(daily, lag) for ref, lag in REFERENCES.items()}
    if first:
        cal["drift"] = {ref: {m: calibrate_drift(z, m, data["plain"], DRIFT_TARGET_FA) for m in DRIFT_METHODS}
                        for ref, z in zs.items()}
    res["drift"] = {ref: drift_metrics(z, data, cal["drift"][ref]) for ref, z in zs.items()}
    res["app"] = app_check(data)
    return res, cal, {"zs": zs, "data": data}


# ---------------------------------------------------------------- report

def describe(name: str) -> str:
    _, area, seed = name.split("_")
    return f"{AREA_KO.get(area, area)} 시드 {seed.lstrip('s')}"


def setup_font():
    for font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if any(font in f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False


def cusum_trace(g: pd.DataFrame, method: str, h: float) -> np.ndarray:
    s, out = 0.0, []
    for z in g[f"z_{method}"].to_numpy():
        s = max(0.0, s + z - CUSUM_K)
        out.append(min(s, h * 1.15))
        if s > h:
            s = 0.0
    return np.array(out)


def plot_example(extras: dict, cal: dict, path: Path):
    """A transient drift over the year: frozen vs lagged reference, weather marks."""
    setup_font()
    drift = extras["data"]["drift"]
    rows = drift[drift["shape"] == "transient"]
    r = rows.iloc[0]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    for ax, ref in zip(axes, ("frozen", "lagged")):
        d = extras["zs"][ref]
        g = d[(d["driver_id"] == r.driver_id) & (d["day_idx"] >= REF_DAYS)]
        x = pd.to_datetime(g["date"])
        h = cal["drift"][ref]["personal_fleet_f"]
        ax.fill_between(x, 0, g["strength"] * h, color="#f2c94c", alpha=0.35, step="mid", label="심은 변화 강도 (정답, 하루 이상 포함)")
        ax.plot(x, cusum_trace(g, "personal_fleet_f", h), color="#1f6feb", lw=1.4, label="누적 이탈 (CUSUM)")
        ax.axhline(h, color="#d64545", ls="--", lw=1, label="경고 기준")
        for w, color in (("rain", "#5b8def"), ("snow", "#9aa5b1")):
            wx = x[g["weather"] == w]
            ax.plot(wx, np.full(len(wx), -0.04 * h), "|", color=color, ms=6, label="비" if w == "rain" else "눈")
        name = "처음 20일 고정 기준" if ref == "frozen" else "최근 기준 (80~20일 전)"
        ax.set_title(f"{r.driver_id}: {'피로 누적' if r.kind == 'fatigue' else '조급해짐'}, 일시적 변화 — {name}")
        ax.legend(fontsize=8, ncol=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_history(res: list, path: Path):
    setup_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    names = {"expanding": "전체 과거", "window60": "최근 60회", "fixed20": "처음 20회 고정"}
    width = 0.27
    for k, (h, label) in enumerate(names.items()):
        rec = np.nanmean([np.array(r["position"][h]["recall"], dtype=float) for r in res], axis=0)
        bins = res[0]["position"][h]["bin"]
        ax.bar(np.arange(len(bins)) + (k - 1) * width, rec, width, label=label)
    labels = [b.replace("[", "").replace(")", "").replace(", ", "~") + "회째" for b in res[0]["position"]["expanding"]["bin"]]
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("하루 이상 탐지율")
    ax.set_xlabel("그 시간대 몇 번째 출퇴근인지")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def pct(x) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f}%"


def ms(vals, fmt="{:.2f}", scale=1.0) -> str:
    v = np.array([np.nan if x is None else x for x in vals], dtype=float) * scale
    if np.all(np.isnan(v)):
        return "-"
    if len(v) == 1:
        return fmt.format(np.nanmean(v))
    return fmt.format(np.nanmean(v)) + " ± " + fmt.format(np.nanstd(v))


def summary(res: list, others: list) -> list:
    mean = lambda f: float(np.nanmean([f(r) for r in others]))
    day = lambda h, k: mean(lambda r: r["day"][h][k])
    fl, rd = "personal_fleet_f", "personal_road"
    dr = lambda ref, k: mean(lambda r: r["drift"][ref][k])
    worst = max(res[1:] or res, key=lambda r: r["drift"]["frozen"][f"fa_{fl}"])
    ch = lambda ev, key: mean(lambda r: r["changes"][ev][key]["after"])
    return [
        "## 요약", "",
        f"- **하루 이상**: 운전자 1명당 연 {day('expanding', f'fa_{fl}'):.1f}회 헛경보에서 하루 이상의 "
        f"{100 * day('expanding', f'recall_{fl}'):.0f}%를 잡는다. 같은 헛경보에서 고정 기준(km당 급정거)은 "
        f"{100 * day('expanding', 'recall_absolute'):.0f}%이다. 주행 특성을 더하면 원래 졸음·거친 성향인 "
        f"운전자의 탐지율이 {100 * day('expanding', f'recall_{rd}_drowsy'):.0f}%/{100 * day('expanding', f'recall_{rd}_aggressive'):.0f}%에서 "
        f"{100 * day('expanding', f'recall_{fl}_drowsy'):.0f}%/{100 * day('expanding', f'recall_{fl}_aggressive'):.0f}%로 올라간다.",
        f"- **비·눈**: 급정거만 보면 비 오는 날 헛경보가 맑은 날의 "
        f"{day('expanding', f'fa100_{rd}_rain') / max(day('expanding', f'fa100_{rd}_clear'), 1e-9):.1f}배이다 "
        "(배경 교통으로 본 도로 사정 보정이 운전자들의 비 효과를 다 반영하지 못함). "
        "'오늘 다른 사용자' 보정을 더하면 맑음·비·눈 헛경보가 "
        f"{day('expanding', f'fa100_{fl}_clear'):.2f}/{day('expanding', f'fa100_{fl}_rain'):.2f}/"
        f"{day('expanding', f'fa100_{fl}_snow'):.2f}회(100회당)로 비슷해진다.",
        f"- **기준선 유지**: 전체 과거와 최근 60회는 차이가 거의 없다. 처음 20회로 고정하면 헛경보는 줄지만 "
        f"탐지율도 {100 * day('fixed20', f'recall_{fl}'):.0f}%로 떨어진다. 앱은 최근 60회 기준으로 충분하다.",
        f"- **이직**: 시간대별로만 기록을 묶으면 새 경로의 첫 30회에서 헛경보가 100회당 {ch('route_change', 'window'):.1f}회로 "
        f"뛴다. 경로·시간대별로 묶어 새 경로에서 기록을 새로 시작하면 {ch('route_change', 'route'):.1f}회이다. "
        f"차량 교체(조금 더 부드러운 운전)는 경고를 늘리지 않는다.",
        f"- **장기 변화**: 하루 값을 다른 사용자 대비로 보정한 CUSUM은 빠른·느린·일시적 변화를 모두 잡고 "
        f"(지연 중앙값 약 {dr('lagged', f'delay_{fl}_fast'):.0f}/{dr('lagged', f'delay_{fl}_slow'):.0f}/"
        f"{dr('lagged', f'delay_{fl}_transient'):.0f}일), 일시적 변화가 끝나면 "
        f"{100 * dr('lagged', f'cleared_{fl}'):.0f}% 조용해진다. "
        f"고정 기준(5일 평균 km당 급정거)은 {100 * dr('lagged', 'detect_absolute_fast'):.0f}%만 잡는다.",
        f"- **고정 기준선의 위험**: 처음 20일만 기준으로 쓰면, 그 기간이 우연히 조용했던 운전자는 1년 내내 경고가 "
        f"난다({describe(worst['run'])}: 헛경보 연 {worst['drift']['frozen'][f'fa_{fl}']:.1f}회, 최근 기준이면 "
        f"{worst['drift']['lagged'][f'fa_{fl}']:.1f}회). 장기 변화도 80~20일 전처럼 계속 갱신되는 기준이 안전하다. "
        "느린 변화(2~3달)도 갱신 기준에서 놓치지 않았다.",
        "- **남은 한계**: 원래 거친 운전자가 조급해지는 경우처럼 평소와 차이가 작은 하루 이상은 여전히 절반 이상 놓친다. "
        "앱 판정의 장기 변화 헛경보는 데이터셋마다 연 0.3~0.7회 정도 흔들린다.",
        ""]


def report(res: list, cal: dict) -> str:
    runs = ", ".join(describe(r["run"]) for r in res)
    others = [r for r in res[1:]] or res
    L = ["# 1년 출퇴근: 개인 기준선이 생활 변화에 조용하고, 잡을 건 잡는가", "",
         f"SUMO 실제 한국 지도, 운전자 100명 × 평일 1년(약 245일) 출퇴근, 데이터셋 {len(res)}개({runs}). "
         "데이터 생성 방식은 `analysis/sumo_sim/life.py` 참고. 날씨(서울 기후 기반 비·눈), 공휴일, 휴가, "
         "출발 시각 ±10분, 매일 컨디션, 이직(새 경로), 차량 교체는 **경고가 나면 안 되는** 변화, "
         "하루 이상(1인당 연 4회)과 장기 변화(30명: 빠른·느린·일시적)는 **잡아야 하는** 변화이다.", "",
         f"경고 기준은 **첫 데이터셋({describe(res[0]['run'])})에서 한 번 정하고** 나머지에 그대로 적용했다 "
         f"(하루 이상: 운전자 1명당 연 {DAY_TARGET_FA:.0f}회 헛경보, 장기 변화: 연 {DRIFT_TARGET_FA}회). "
         f"표의 값은 나머지 데이터셋 {len(others)}개의 평균(± 표준편차)이다. 헛경보는 아무 변화도 없는 "
         "운전자(장기 변화·이직·차량 교체 없음)의 평범한 출퇴근에서 센 것이다.", ""]
    L += summary(res, others)
    L += ["## 1. 하루 이상", "",
         "| 방법 | 헛경보 (1인·연) | 탐지율 | 정상 운전자 | 졸음 성향 | 거친 성향 | 피로 | 조급 |",
         "|---|---|---|---|---|---|---|---|"]
    for m in DAY_METHODS:
        g = lambda k: [r["day"]["expanding"][k] for r in others]
        L.append(f"| {METHOD_KO.get(m, m)} | {ms(g(f'fa_{m}'), '{:.1f}')} | {ms(g(f'recall_{m}'), '{:.0f}%', 100)} | "
                 + " | ".join(ms(g(f"recall_{m}_{x}"), "{:.0f}%", 100)
                              for x in ("normal", "drowsy", "aggressive", "fatigue", "hurried")) + " |")
    L += ["", "### 비·눈 오는 날 헛경보 (평범한 출퇴근 100회당)", "",
          "| 방법 | 맑음 | 비 | 눈 |", "|---|---|---|---|"]
    for m in DAY_METHODS:
        g = lambda w: [r["day"]["expanding"][f"fa100_{m}_{w}"] for r in others]
        L.append(f"| {METHOD_KO.get(m, m)} | " + " | ".join(ms(g(w), "{:.2f}") for w in ("clear", "rain", "snow")) + " |")
    L += ["", "### 기준선을 어떻게 유지할까 (개인 + 주행 특성 + 오늘 다른 사용자 보정)", "",
          "| 기준선 | 헛경보 (1인·연) | 탐지율 |", "|---|---|---|"]
    names = {"expanding": "전체 과거 출퇴근", "window60": "최근 60회", "fixed20": "처음 20회로 고정"}
    for h, label in names.items():
        g = lambda k: [r["day"][h][k] for r in others]
        L.append(f"| {label} | {ms(g('fa_personal_fleet_f'), '{:.1f}')} | {ms(g('recall_personal_fleet_f'), '{:.0f}%', 100)} |")
    L += ["", "![history](../images/personal_year_history.png)", "",
          "### 이직·차량 교체 뒤 첫 30회 출퇴근의 헛경보 (100회당)", "",
          "| 사건 | 기록 묶음 | 변화 전 | 변화 후 |", "|---|---|---|---|"]
    for ev, ko in (("route_change", "이직 (새 경로)"), ("vehicle_change", "차량 교체")):
        for key, kko in (("window", "시간대별"), ("route", "경로·시간대별")):
            g = lambda k: [r["changes"][ev][key][k] for r in others]
            L.append(f"| {ko} | {kko} | {ms(g('before'), '{:.2f}')} | {ms(g('after'), '{:.2f}')} |")

    L += ["", "## 2. 장기 변화", "",
          f"하루씩의 이탈(z)을 누적하는 CUSUM이다. 하루 값은 ±{DAY_Z_CAP:.0f}으로 잘라서 하루 이상 한 번으로는 "
          "경고가 나지 않게 했고, 경고가 나면 0부터 다시 쌓는다. 기준은 두 가지이다: "
          f"**고정**(처음 {REF_DAYS}일), **최근**({LAG_GAP + LAG_LEN}~{LAG_GAP}일 전, 계속 갱신).", "",
          "| 기준 | 방법 | 헛경보 (1인·연) | 빠른 변화 탐지 (지연) | 느린 변화 탐지 (지연) | 일시적 변화 탐지 (지연) | "
          "일시적 변화 끝난 뒤 조용해짐 | 이직 뒤 60일 경고 | 차량 교체 뒤 60일 경고 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for ref, rko in (("frozen", "고정"), ("lagged", "최근")):
        for m in DRIFT_METHODS:
            g = lambda k: [r["drift"][ref].get(k) for r in others]
            cells = [ms(g(f"fa_{m}"), "{:.2f}")]
            for shape in ("fast", "slow", "transient"):
                cells.append(f"{ms(g(f'detect_{m}_{shape}'), '{:.0f}%', 100)} ({ms(g(f'delay_{m}_{shape}'), '{:.0f}')}일)")
            cells.append(ms(g(f"cleared_{m}"), "{:.0f}%", 100))
            cells.append(ms(g(f"route_change_{m}"), "{:.0f}%", 100))
            cells.append(ms(g(f"vehicle_change_{m}"), "{:.0f}%", 100))
            L.append(f"| {rko} | {METHOD_KO.get(m, m) if m != 'absolute' else '고정 기준 (5일 평균 km당 급정거)'} | "
                     + " | ".join(cells) + " |")
    ch = np.mean([r["drift"]["lagged"]["chance_after"]["personal_fleet_f"] for r in others])
    L += ["", f"(이직 뒤 60일 경고는 비교 기준이 필요하다: 변화 없는 운전자도 최근 기준 + 다른 사용자 보정에서 "
              f"아무 60일 동안 경고가 날 확률이 약 {100 * ch:.0f}%이다.)", "",
          "![example](../images/personal_year_example.png)", "",
          "## 3. 앱 설정 확인", "",
          "앱(`PersonalBaseline.kt`)의 판정을 그대로 옮겨 1년 데이터에 돌렸다. 하루 판정은 **같은 경로·시간대의 최근 "
          f"{APP_HISTORY}회**만 기록으로 쓴다(새 경로는 기록을 새로 모은다). 장기 변화는 시간대·경로별로, 80~20일 전 기준, "
          f"하루 값 ±{DAY_Z_CAP:.0f} 제한, 경고 뒤 다시 쌓기로 판정한다. 서버의 오늘/평소 도로 기대값 비가 "
          f"{APP_ROAD_SKIP} 미만인 날(눈: 모두가 천천히 달려 길이 유난히 순한 날)은 장기 판정에서 뺀다. 눈 오는 1~2월이 "
          f"기준 기간에 들어가면 봄의 평범한 날이 나빠 보여서, 3~5월에 헛경보가 몰렸기 때문이다. "
          f"헛경보는 운전자가 보는 알림 수(조용한 {APP_RECENT}일 뒤의 경보)로 세고, 경고 기준 H = {APP_DRIFT_H:.1f}은 "
          f"다섯 데이터셋을 합쳐 연 {DRIFT_TARGET_FA}회가 되게 정했다(데이터셋 하나에는 변화 없는 운전자가 50명뿐이라 "
          f"하나로 정하면 흔들린다). '오늘 다른 사용자' 보정은 넣지 않았다. 각 차의 이탈 값을 서버에 "
          "보내야 하는데, 앱은 개인 기록을 차 밖으로 보내지 않는다. 갱신 기준에서는 이 보정이 없어도 결과가 거의 같다(2장).", "",
          "급정거 검정은 포아송 대신 **그 운전자의 날마다 퍼짐**(과거 급정거의 분산/평균, 10회 분량으로 1 쪽으로 축소)을 쓴다. "
          "원래 급정거가 잦은 운전자는 매일 비슷한 횟수로 급정거해서(분산/평균 약 0.2~0.35, 정상 운전자는 약 1), 포아송으로는 "
          "평범해 보이는 증가도 그 사람에게는 이상한 경우가 있다. 같은 헛경보에서 비교하면 거친 성향 운전자의 조급한 날 탐지가 "
          "32% → 39%, 졸음 성향의 조급한 날이 57% → 64%, 졸음 성향의 피곤한 날이 17% → 21%로 올랐고 정상 운전자는 그대로였다.", "",
          "| 판정 | 헛경보 (1인·연) | 탐지율 |", "|---|---|---|"]
    g = lambda k: [r["app"][k] for r in others]
    L.append(f"| 하루: 주의 (p < 10⁻³) | {ms(g('day_fa_watch'), '{:.1f}')} | {ms(g('day_recall_watch'), '{:.0f}%', 100)} |")
    L.append(f"| 하루: 경고 (p < 10⁻⁴) | {ms(g('day_fa_alert'), '{:.1f}')} | {ms(g('day_recall_alert'), '{:.0f}%', 100)} |")
    L.append(f"| 장기 변화 | {ms(g('drift_fa'), '{:.2f}')} | 빠른 {ms(g('drift_detect_fast'), '{:.0f}%', 100)}"
             f" ({ms(g('drift_delay_fast'), '{:.0f}')}일), 느린 {ms(g('drift_detect_slow'), '{:.0f}%', 100)}"
             f" ({ms(g('drift_delay_slow'), '{:.0f}')}일), 일시적 {ms(g('drift_detect_transient'), '{:.0f}%', 100)}"
             f" ({ms(g('drift_delay_transient'), '{:.0f}')}일) |")
    L += ["", f"**피곤함·서두름 문구:** 경고가 난 심은 이상한 날에서 '가속도 RMS가 중간 감속보다 더 크게 벗어났으면 서두름, "
          f"아니면 피곤함' 규칙이 맞은 비율은 {ms(g('cause_acc'), '{:.0f}%', 100)}이다(피곤한 날 "
          f"{ms(g('cause_acc_fatigue'), '{:.0f}%', 100)}, 서두른 날 {ms(g('cause_acc_hurried'), '{:.0f}%', 100)}). "
          "평균 속도 같은 다른 특성을 더한 규칙은 오히려 낮았다."]
    L += ["", "경고(p < 10⁻⁴)의 운전자 성향별 탐지율: 정상 "
          + ms(g("day_recall_alert_normal"), "{:.0f}%", 100) + ", 졸음 성향 "
          + ms(g("day_recall_alert_drowsy"), "{:.0f}%", 100) + ", 거친 성향 "
          + ms(g("day_recall_alert_aggressive"), "{:.0f}%", 100) + "."]
    L.append("")
    return "\n".join(L)


def main():
    folders = sorted((p for p in OUT.glob("y_*") if (p / "trips.csv").exists()),
                     key=lambda p: (p.name != "y_gangnam_s3", p.name))
    res, cal, first_extras = [], None, None
    for folder in folders:
        print(f"== {folder.name}", flush=True)
        r, cal, extras = evaluate(folder, cal)
        if first_extras is None:
            first_extras = extras
        res.append(r)
        (folder / "year_metrics.json").write_text(json.dumps(r, ensure_ascii=False, default=float, indent=1),
                                                   encoding="utf-8")
    (OUT / "year_calibration.json").write_text(json.dumps(cal, indent=1), encoding="utf-8")
    plot_example(first_extras, cal, DOCS / "images" / "personal_year_example.png")
    plot_history(res[1:] or res, DOCS / "images" / "personal_year_history.png")
    md = report(res, cal)
    (DOCS / "results" / "personal_year.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
