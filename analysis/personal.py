"""Personal commute baseline: anomalies in the SAME driver's driving on the SAME commute
(same route, same time window) compared with their own history.

Two questions
  1. One-day anomaly: is today's commute unusual *for this driver* (tired, rushed day)?
  2. Long-term drift: has this driver's driving been changing over weeks, and since when?

Per commute trip (SUMO trips.csv, or app logs in the same format):
  hard_brakes, moderate_brakes, accel_rms, max_decel, distance_m, plus its road traversals.

Expected hard brakes of a trip
  E_usual : sum over its traversals of the segment x time-band rate on other days
            (baseline.cell_baseline) -> "how brake-prone this route at this time usually is"
  E_today : same, but from OTHER vehicles on the same segments TODAY
            -> "how today's road conditions are" (congestion, incidents)

Methods compared (all past-only, as they would run in the car)
  absolute         hard brakes per km, one threshold for everyone (TMAP-style)
  population       hard brakes vs other drivers on the same route today (E_today)
  personal         vs the driver's own past commutes on this route/window (f_driver x E_usual)
  personal_road    personal, adjusted for today's road conditions (f_driver x E_today)   [proposed]
  personal_road_f  personal_road + moderate braking, accel RMS, max decel vs own history [proposed+]

    python personal.py out/personal/p_gangnam_s3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from scipy.stats import t as t_dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baseline import add_context, cell_baseline, poisson_upper_p  # noqa: E402

K_ROAD = 20.0      # pseudo-passes shrinking today's road rate toward the usual rate
A_PERSONAL = 2.0   # pseudo expected hard brakes shrinking a driver's factor toward 1 (population)
N0_FEATURE = 3.0   # pseudo trips shrinking a driver's feature variance toward the population's
Z_CAP = 4.0        # cap on a feature's z so one heavy-tailed feature can't decide alone
MIN_HISTORY = 3    # past commutes on the same route/window needed before scoring
CALIB_DAYS = 5     # first weekdays used as each driver's reference period for drift
FEATURES = ["moderate_per_km", "accel_rms", "max_decel"]
METHODS = ["absolute", "population", "personal", "personal_road", "personal_road_f"]
METHOD_KO = {
    "absolute": "고정 기준 (km당 급정거)",
    "population": "다른 운전자 대비 (같은 길·오늘)",
    "personal": "개인 기준선",
    "personal_road": "개인 기준선 + 도로 사정 보정",
    "personal_road_f": "개인 + 도로 보정 + 주행 특성",
    "personal_fleet_f": "개인 + 주행 특성 + 오늘 다른 사용자 보정",
}


# ---------------------------------------------------------------- per-trip expectations

def trip_expectations(trav: pd.DataFrame) -> pd.DataFrame:
    """E_usual and E_today (expected hard brakes) for every named-driver trip."""
    df = add_context(trav)
    return expectations_from(df[df["driver_id"] != "bg"], cell_baseline(df))


def expectations_from(own: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """own: named-driver traversals (trip_id, hard_brakes, segment_id, bucket, date);
    cells: cell_baseline / cell_rates output."""
    keys = ["segment_id", "bucket", "date"]
    cells = cells[keys + ["exposure", "count", "base_rate"]]
    own = own[["trip_id", "hard_brakes"] + keys].merge(cells, on=keys)
    lam = own["base_rate"]
    others_n = (own["exposure"] - 1).clip(lower=0)
    others_y = (own["count"] - own["hard_brakes"]).clip(lower=0)
    own["e_usual"] = lam
    own["e_today"] = (others_y + K_ROAD * lam) / (others_n + K_ROAD)
    return own.groupby("trip_id")[["e_usual", "e_today"]].sum().reset_index()


def load(folder: Path) -> pd.DataFrame:
    trips = pd.read_csv(folder / "trips.csv")
    trips = trips[(trips["trip_kind"] == "commute") & (trips["distance_m"] > 300)].copy()
    # SUMO artifacts (~1 in 10,000 trips): a collision shows up as an impossible deceleration and a
    # vehicle lost mid-trip as garbage values. Drop corrupt rows; cap deceleration at a physical 10 m/s².
    ok = (trips["distance_m"] < 1e6) & (trips["accel_rms"].between(0, 10)) & (trips["max_decel"] < 1e3)
    trips = trips[ok].copy()
    trips["max_decel"] = trips["max_decel"].clip(upper=10.0)
    exp = trip_expectations(pd.read_csv(folder / "traversals.csv"))
    t = trips.merge(exp, on="trip_id", how="inner")
    t["km"] = t["distance_m"] / 1000
    t["moderate_per_km"] = t["moderate_brakes"] / t["km"]
    t["hard_per_km"] = t["hard_brakes"] / t["km"]
    return t.sort_values(["driver_id", "window", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------- one-day anomalies

def _past_sum(g, col: str, history) -> pd.Series:
    """Sum of col over each trip's past trips in its group (rows in time order).
    history None: all past trips; ("window", n): the last n; ("fixed", n): the group's first n only."""
    if history is None:
        f = lambda s: s.shift().cumsum()
    elif history[0] == "window":
        f = lambda s: s.shift().rolling(history[1], min_periods=1).sum()
    else:
        f = lambda s: s.where(np.arange(len(s)) < history[1], 0.0).shift().cumsum()
    return g[col].transform(f).fillna(0.0)


N0_DISP = 10.0       # pseudo trips shrinking the spread toward Poisson (variance = mean)


def _dispersed_p(t, prev, y, mu, p_poisson):
    """Hard-brake p-value with each driver's own variance/mean ratio (past trips, shrunk to 1).
    Drivers who brake hard often do so very regularly (ratio ~0.3), so a rise that Poisson
    calls ordinary can be unusual for them. Normal approximation where the expectation is >= 1."""
    t["_y2"] = t["hard_brakes"] ** 2
    n = t["n_prev"].clip(lower=1)
    m = prev("hard_brakes") / n
    v = (prev("_y2") / n - m**2).clip(lower=0) * n / (n - 1).clip(lower=1)
    phi = (t["n_prev"] * (v / m.clip(lower=0.2)) + N0_DISP) / (t["n_prev"] + N0_DISP)
    phi = phi.clip(0.25, 3.0).to_numpy()
    t.drop(columns="_y2", inplace=True)
    z = (y - 0.5 - mu) / np.sqrt(phi * mu + 0.05)
    return np.where(mu >= 1.0, norm.sf(z), p_poisson)


def score_trips(t: pd.DataFrame, keys=("driver_id", "window"), history=None, dispersion=False,
                pop_var: dict | None = None) -> pd.DataFrame:
    """Anomaly scores (higher = more unusual) of every method; past-only personal history.
    keys: what counts as 'the same commute' (add a route column to restart history on a new route).
    history: which past trips form the baseline (see _past_sum).
    dispersion: hard-brake test with the driver's own day-to-day spread (_dispersed_p), as the app.
    pop_var: fixed population variances of the features (the app's constants) instead of the data's."""
    t = t.copy()
    keys = list(keys)
    g = t.groupby(keys, sort=False)
    prev = lambda col: _past_sum(g, col, history)
    t["pos"] = g.cumcount()
    t["n_prev"] = t["pos"] if history is None else t["pos"].clip(upper=history[1])
    y_prev = prev("hard_brakes")
    t["f_usual"] = (y_prev + A_PERSONAL) / (prev("e_usual") + A_PERSONAL)
    t["f_today"] = (y_prev + A_PERSONAL) / (prev("e_today") + A_PERSONAL)

    y = t["hard_brakes"].to_numpy()
    p_pop = poisson_upper_p(y, t["e_today"].to_numpy())
    p_personal = poisson_upper_p(y, (t["f_usual"] * t["e_usual"]).to_numpy())
    p_road = poisson_upper_p(y, (t["f_today"] * t["e_today"]).to_numpy())
    if dispersion:
        p_road = _dispersed_p(t, prev, y, (t["f_today"] * t["e_today"]).to_numpy(), p_road)

    # continuous features vs own past trips (variance shrunk toward the population's)
    feature_p, feature_z = [], []
    for col in FEATURES:
        x = t[col]
        t["_sq"] = x**2
        s1, s2 = prev(col), prev("_sq")
        n = t["n_prev"].clip(lower=1)
        mean = s1 / n
        var = (s2 / n - mean**2).clip(lower=0)
        pv = pop_var[col] if pop_var else t.groupby(keys)[col].var().median()
        var = (t["n_prev"] * var + N0_FEATURE * pv) / (t["n_prev"] + N0_FEATURE)
        # predictive t: with few past trips the mean and variance are themselves uncertain, and
        # z is capped so one heavy-tailed feature (one sharp stop -> max_decel) can't dominate
        z = ((x - mean) / np.sqrt(var.clip(lower=1e-9) * (1 + 1 / n))).clip(upper=Z_CAP)
        feature_z.append(z)
        feature_p.append(t_dist.sf(z, t["n_prev"] + N0_FEATURE))
    t = t.drop(columns="_sq")
    fisher = lambda ps: chi2.sf(-2 * sum(np.log(np.clip(p, 1e-300, 1)) for p in ps), 2 * len(ps))
    p_feat = fisher([p_road] + feature_p)

    # fleet: how every app user drove on this commute window today (weather, citywide
    # conditions). Hard brakes vs each user's personal expectation, trimmed so today's few
    # anomalous drivers don't set it; feature z-scores shifted by today's median z.
    day = t["date"].astype(str) + t["window"].astype(str)
    exp_p = t["f_usual"] * t["e_usual"]
    ratio = (t["hard_brakes"] + 1) / (exp_p + 1)
    keep = ratio <= ratio.groupby(day).transform(lambda s: s.quantile(0.75))
    f_raw = (t["hard_brakes"] * keep).groupby(day).transform("sum") / (exp_p * keep).groupby(day).transform(
        "sum").clip(lower=1e-9)
    t["fleet"] = f_raw / f_raw.median()
    p_fleet = poisson_upper_p(y, (exp_p * t["fleet"]).to_numpy())
    fleet_feat = [t_dist.sf(z - z.groupby(day).transform("median"), t["n_prev"] + N0_FEATURE) for z in feature_z]
    p_fleet_f = fisher([p_fleet] + fleet_feat)

    t["s_absolute"] = t["hard_per_km"]
    t["s_population"] = -np.log(np.clip(p_pop, 1e-300, 1))
    t["s_personal"] = -np.log(np.clip(p_personal, 1e-300, 1))
    t["s_personal_road"] = -np.log(np.clip(p_road, 1e-300, 1))
    t["s_personal_road_f"] = -np.log(np.clip(p_feat, 1e-300, 1))
    t["s_personal_fleet_f"] = -np.log(np.clip(p_fleet_f, 1e-300, 1))
    return t


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def evaluate_day(t: pd.DataFrame, day_truth: pd.DataFrame, drift_ids: set) -> tuple[dict, pd.DataFrame]:
    keyset = set(zip(day_truth["driver_id"], day_truth["date"], day_truth["window"]))
    kind_of = {(r.driver_id, r.date, r.window): r.kind for r in day_truth.itertuples()}
    e = t[(t["n_prev"] >= MIN_HISTORY) & ~t["driver_id"].isin(drift_ids)].copy()
    e["anomaly"] = [k in keyset for k in zip(e["driver_id"], e["date"], e["window"])]
    e["kind"] = [kind_of.get(k, "") for k in zip(e["driver_id"], e["date"], e["window"])]
    k = int(e["anomaly"].sum())
    res = {"n_trips": len(e), "n_anomalies": k}
    for m in METHODS:
        s = e[f"s_{m}"].to_numpy()
        res[f"auc_{m}"] = auc(s, e["anomaly"].to_numpy())
        top = e.nlargest(k, f"s_{m}")
        res[f"p@k_{m}"] = top["anomaly"].mean()
        for label in ("normal", "drowsy", "aggressive"):
            grp = e[e["label"] == label]
            res[f"auc_{m}_{label}"] = auc(grp[f"s_{m}"].to_numpy(), grp["anomaly"].to_numpy())
            # share of this group's anomalies found in the overall top-K
            res[f"recall_{m}_{label}"] = top[(top["label"] == label)]["anomaly"].sum() / max(
                (grp["anomaly"]).sum(), 1)
        for kind in ("fatigue", "hurried"):
            grp = e[(e["kind"] == kind) | ~e["anomaly"]]
            res[f"auc_{m}_{kind}"] = auc(grp[f"s_{m}"].to_numpy(), grp["anomaly"].to_numpy())
    return res, e


# ---------------------------------------------------------------- long-term drift

def daily_series(t: pd.DataFrame) -> pd.DataFrame:
    """Per driver x weekday: observed hard brakes, expectations and feature means."""
    agg = {"hard_brakes": "sum", "e_usual": "sum", "e_today": "sum", "km": "sum",
           "moderate_brakes": "sum", "accel_rms": "mean", "max_decel": "mean"}
    d = t.groupby(["driver_id", "label", "date"]).agg(agg).reset_index().sort_values(["driver_id", "date"])
    d["moderate_per_km"] = d["moderate_brakes"] / d["km"]
    d["hard_per_km"] = d["hard_brakes"] / d["km"]
    d["day_idx"] = d.groupby("driver_id").cumcount()
    return d


def drift_statistics(d: pd.DataFrame) -> pd.DataFrame:
    """Daily z-scores of every method vs each driver's first CALIB_DAYS weekdays (frozen)."""
    d = d.copy()
    cal = d[d["day_idx"] < CALIB_DAYS].groupby("driver_id")
    ref = pd.DataFrame({
        "y": cal["hard_brakes"].sum(), "eu": cal["e_usual"].sum(), "et": cal["e_today"].sum(),
        **{f"m_{c}": cal[c].mean() for c in FEATURES + ["hard_per_km"]},
        **{f"v_{c}": cal[c].var() for c in FEATURES},
    })
    pop_var = {c: ref[f"v_{c}"].median() for c in FEATURES}
    d = d.merge(ref, left_on="driver_id", right_index=True)
    f_usual = (d["y"] + A_PERSONAL) / (d["eu"] + A_PERSONAL)
    f_today = (d["y"] + A_PERSONAL) / (d["et"] + A_PERSONAL)
    pz = lambda obs, exp: (obs - exp) / np.sqrt(exp + 0.25)
    d["z_absolute"] = d["hard_per_km"]                      # handled by rolling threshold
    d["z_population"] = pz(d["hard_brakes"], d["e_today"])
    d["z_personal"] = pz(d["hard_brakes"], f_usual * d["e_usual"])
    d["z_personal_road"] = pz(d["hard_brakes"], f_today * d["e_today"])
    zf = [(d[c] - d[f"m_{c}"]) / np.sqrt(((CALIB_DAYS * d[f"v_{c}"].fillna(pop_var[c]) + N0_FEATURE * pop_var[c])
                                          / (CALIB_DAYS + N0_FEATURE)).clip(lower=1e-9)) for c in FEATURES]
    d["z_personal_road_f"] = (d["z_personal_road"] + sum(zf)) / np.sqrt(1 + len(FEATURES))
    return d


def first_alarm(d: pd.DataFrame, method: str, threshold: float, k: float = 0.5):
    """Driver -> (first alarm day_idx or None, max statistic) after the reference period."""
    out = {}
    for did, g in d.groupby("driver_id"):
        g = g[g["day_idx"] >= CALIB_DAYS]
        if method == "absolute":   # rolling 5-weekday hard brakes per km above a fixed threshold
            stat = g["hard_per_km"].rolling(5, min_periods=3).mean().fillna(0).to_numpy()
        else:                      # one-sided CUSUM
            s, stat = 0.0, []
            for z in g[f"z_{method}"].to_numpy():
                s = max(0.0, s + z - k)
                stat.append(s)
            stat = np.array(stat)
        hit = np.nonzero(stat > threshold)[0]
        out[did] = (int(g["day_idx"].iloc[hit[0]]) if len(hit) else None, stat.max() if len(stat) else 0.0)
    return out


def evaluate_drift(d: pd.DataFrame, drift_truth: pd.DataFrame, false_alarm: float = 0.10) -> dict:
    start_idx = {}
    for r in drift_truth.itertuples():
        days = d[(d["driver_id"] == r.driver_id)].set_index("date")["day_idx"]
        later = days[days.index >= r.start_date]
        start_idx[r.driver_id] = int(later.iloc[0]) if len(later) else None
    drift_ids = set(start_idx)
    res = {"n_drift": len(drift_ids), "false_alarm_target": false_alarm}
    for m in METHODS:
        maxes = first_alarm(d, m, threshold=np.inf)
        null = [v[1] for did, v in maxes.items() if did not in drift_ids]
        thr = float(np.quantile(null, 1 - false_alarm)) if null else np.inf
        alarms = first_alarm(d, m, thr)
        detected, delays, early = 0, [], 0
        for did, s in start_idx.items():
            a = alarms.get(did, (None, 0))[0]
            if a is None or s is None:
                continue
            if a < s:
                early += 1
            else:
                detected += 1
                delays.append(a - s)
        res[f"detect_{m}"] = detected / max(len(drift_ids), 1)
        res[f"delay_{m}"] = float(np.median(delays)) if delays else float("nan")
        res[f"early_{m}"] = early
        res[f"threshold_{m}"] = thr
    return res


# ---------------------------------------------------------------- report

def run(folder: Path) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    t = load(folder)
    labels = pd.read_csv(folder / "truth_drivers.csv")[["driver_id", "label"]]
    t = t.merge(labels, on="driver_id")
    day_truth = pd.read_csv(folder / "truth_day_anomalies.csv")
    drift_truth = pd.read_csv(folder / "truth_drift.csv")
    scored = score_trips(t)
    day_res, day_eval = evaluate_day(scored, day_truth, set(drift_truth["driver_id"]))
    daily = drift_statistics(daily_series(t))
    drift_res = evaluate_drift(daily, drift_truth)
    return day_res, drift_res, day_eval, daily


def plot_example(daily: pd.DataFrame, drift_truth: pd.DataFrame, path: Path):
    for font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if any(font in f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, kind in zip(axes, ("fatigue", "hurried")):
        rows = drift_truth[drift_truth["kind"] == kind]
        if rows.empty:
            continue
        r = rows.iloc[0]
        g = daily[daily["driver_id"] == r.driver_id]
        g = g[g["day_idx"] >= CALIB_DAYS]
        for m, color in (("population", "#9aa5b1"), ("personal_road", "#1f6feb"), ("personal_road_f", "#d64545")):
            s, cus = 0.0, []
            for z in g[f"z_{m}"]:
                s = max(0.0, s + z - 0.5)
                cus.append(s)
            ax.plot(pd.to_datetime(g["date"]), cus, "-o", ms=3, color=color, label=METHOD_KO[m])
        ax.axvline(pd.to_datetime(r.start_date), color="#555", ls="--", lw=1, label="변화 시작 (정답)")
        ax.set_title(f"장기 변화 예시: {r.driver_id} ({'피로 누적' if kind == 'fatigue' else '조급해짐'})")
        ax.set_ylabel("누적 이탈 (CUSUM)")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    args = ap.parse_args()
    day, drift, _, daily = run(args.folder)
    print(pd.Series(day).round(3).to_string())
    print(pd.Series(drift).round(3).to_string())
    plot_example(daily, pd.read_csv(args.folder / "truth_drift.csv"), args.folder / "drift_example.png")


if __name__ == "__main__":
    main()
