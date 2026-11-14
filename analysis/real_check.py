"""Checks the personal baseline on REAL commute records from the car, the moment they exist.

The app keeps one line per commute on the device (files/personal/commutes.csv):
    date, window, route, km, hard_brakes, moderate_brakes, accel_rms, max_decel, e_usual, e_today
Pull it per car / driver:
    adb exec-out run-as com.drivepulse.app --user 10 cat files/personal/commutes.csv > car01.csv

    python real_check.py car01.csv car02.csv [--labels labels.csv] [--out docs/results/real_check.md]

labels.csv (optional): days the driver says were unusual, e.g. from a short note after the trip
    driver_id, date, window, note        (driver_id = file name without .csv)

What it reports
  1. how much data there is (commutes per route and window)
  2. day-to-day variation of ordinary commutes vs the SUMO simulation the app was tuned on
  3. what the app would have reported each day (same rules as PersonalBaseline.kt)
  4. with labels: how many of the unusual days the app caught, how many alerts were right
  5. suggested re-calibration: feature population variances and the alert p-value for a
     target number of alerts per driver-year

Try it without a car (SUMO commutes in the app's format, labels from the planted anomalies):
    python real_check.py --from-sim out/year/y_gangnam_s4 --drivers 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from personal import FEATURES, MIN_HISTORY, score_trips  # noqa: E402
from personal_year import (APP_DRIFT_H, APP_HISTORY, APP_P, APP_POP_VAR, REFERENCES, YEAR_DAYS,  # noqa: E402
                           app_notifications, app_units, drift_z)

# PersonalBaseline.kt fallbacks
DEFAULT_RATE_PER_KM = 0.15
MAX_DECEL_CAP = 10.0
# ordinary commutes in the 1-year SUMO data (clear days, median over driver x window)
SIM = {"hard_disp": 0.9, "accel_cv": 0.13, "alerts_per_year": 2.4, "drift_per_year": 0.6}
TARGET_ALERTS = 2.0       # per driver-year, for the suggested alert threshold
MIN_GROUP = 10            # commutes on a route/window before its variation is used


def load_commutes(paths: list[Path]) -> pd.DataFrame:
    parts = []
    for p in paths:
        d = pd.read_csv(p, dtype={"route": str, "date": str, "window": str})
        if "driver_id" not in d.columns:
            d["driver_id"] = p.stem
        parts.append(d)
    t = pd.concat(parts, ignore_index=True)
    t = t[t["window"].isin(["am", "pm"]) & (t["km"] > 0.3)].copy()
    t["max_decel"] = t["max_decel"].clip(upper=MAX_DECEL_CAP)
    # the app's fallbacks when the server was unreachable
    t["e_today"] = t["e_today"].fillna(t["e_usual"]).fillna(t["km"] * DEFAULT_RATE_PER_KM)
    t["e_usual"] = t["e_usual"].fillna(t["e_today"])
    t["moderate_per_km"] = t["moderate_brakes"] / t["km"]
    t["hard_per_km"] = t["hard_brakes"] / t["km"]
    t["label"], t["weather"], t["strength"] = "unknown", "", 0.0
    return t.sort_values(["driver_id", "window", "date"]).reset_index(drop=True)


def attach_labels(t: pd.DataFrame, labels: Path | None) -> pd.DataFrame:
    t["anomaly"] = False
    if labels is not None:
        lab = pd.read_csv(labels, dtype=str)
        keys = set(zip(lab["driver_id"], lab["date"], lab["window"]))
        t["anomaly"] = [k in keys for k in zip(t["driver_id"], t["date"], t["window"])]
    return t


def variation(t: pd.DataFrame) -> dict:
    """Day-to-day variation of ordinary commutes per driver x window x route (median)."""
    g = t[~t["anomaly"]].groupby(["driver_id", "window", "route"])
    big = g.size()
    big = big[big >= MIN_GROUP].index
    if len(big) == 0:
        return {"groups": 0}
    rows = []
    for key in big:
        x = g.get_group(key)
        m = x["hard_brakes"].mean()
        rows.append({"hard_disp": x["hard_brakes"].var() / m if m >= 0.2 else np.nan,
                     "accel_cv": x["accel_rms"].std() / x["accel_rms"].mean(),
                     **{f"var_{c}": x[c].var() for c in FEATURES}})
    r = pd.DataFrame(rows).median()
    return {"groups": len(big), **r.to_dict()}


def replay(t: pd.DataFrame) -> pd.DataFrame:
    """The app's one-day verdict for every commute (LEARNING / NORMAL / WATCH / ALERT)."""
    s = score_trips(t, keys=("driver_id", "window", "route"), history=("window", APP_HISTORY),
                    dispersion=True, pop_var=APP_POP_VAR)
    p = np.exp(-s["s_personal_road_f"])
    s["p"] = p
    s["level"] = np.where(s["n_prev"] < MIN_HISTORY, "LEARNING",
                          np.where(p < APP_P["alert"], "ALERT", np.where(p < APP_P["watch"], "WATCH", "NORMAL")))
    return s


def drift_notes(t: pd.DataFrame) -> dict:
    u = app_units(t)
    if u.empty:
        return {}
    nt = app_notifications(drift_z(u, REFERENCES["lagged"]), APP_DRIFT_H)
    out = {}
    for unit, dates in nt.items():
        if dates:
            out.setdefault(unit.split("|")[0], []).extend(dates)
    return out


def driver_years(t: pd.DataFrame) -> pd.Series:
    """Commute days per driver / workdays in a year."""
    return t.groupby("driver_id")["date"].nunique() / YEAR_DAYS


def suggest(s: pd.DataFrame, years: pd.Series) -> dict:
    e = s[(s["n_prev"] >= MIN_HISTORY) & ~s["anomaly"]]
    total_years = years.sum()
    k = int(round(TARGET_ALERTS * total_years))
    p_sorted = np.sort(e["p"].to_numpy())
    p_thr = float(p_sorted[k]) if 0 < k < len(p_sorted) else float("nan")
    return {"p_alert_for_target": p_thr, "commutes_scored": len(e), "driver_years": float(total_years)}


def report(t: pd.DataFrame, s: pd.DataFrame, drift: dict, var: dict, sug: dict, has_labels: bool) -> str:
    years = driver_years(t)
    per = s.groupby("driver_id")["level"].value_counts().unstack(fill_value=0)
    for c in ("LEARNING", "NORMAL", "WATCH", "ALERT"):
        if c not in per.columns:
            per[c] = 0
    L = ["# 실제 출퇴근 기록 점검", "",
         f"운전자 {t['driver_id'].nunique()}명, 출퇴근 {len(t)}회, {t['date'].min()} ~ {t['date'].max()}.", "",
         "## 1. 데이터", "",
         "| 운전자 | 출퇴근 일수 | 경로·시간대 묶음 (10회 이상) | 가장 많은 묶음 |", "|---|---|---|---|"]
    for did, g in t.groupby("driver_id"):
        c = g.groupby(["window", "route"]).size()
        L.append(f"| {did} | {g['date'].nunique()} | {len(c)} ({int((c >= MIN_GROUP).sum())}) | {int(c.max())}회 |")
    L += ["", "판정에는 같은 경로·시간대 기록이 3회 이상 필요하고, 장기 판정은 23회부터 시작한다.", "",
          "## 2. 평소 출퇴근의 날마다 변동 (시뮬레이션과 비교)", ""]
    if var.get("groups", 0) == 0:
        L.append(f"같은 경로·시간대에 {MIN_GROUP}회 이상 쌓인 묶음이 아직 없다.")
    else:
        L += [f"같은 경로·시간대 {MIN_GROUP}회 이상인 묶음 {var['groups']}개의 중앙값이다.", "",
              "| 항목 | 실제 | 시뮬레이션 / 앱 값 | 뜻 |", "|---|---|---|---|",
              f"| 급정거 분산/평균 | {var['hard_disp']:.2f} | {SIM['hard_disp']} | 1보다 크면 실제가 더 들쭉날쭉 |",
              f"| 가속도 RMS 변동계수 | {var['accel_cv']:.3f} | {SIM['accel_cv']} | |",
              *[f"| {c} 분산 | {var[f'var_{c}']:.4f} | {APP_POP_VAR[c]} (앱 상수) | 앱이 기록이 적을 때 쓰는 값 |"
                for c in FEATURES]]
    L += ["", "## 3. 앱이 냈을 판정", "",
          "| 운전자 | 학습 중 | 평소와 비슷 | 주의 | 경고 | 경고 (연 환산) | 표시 없는 날 경고 (연 환산) | 장기 변화 알림 |",
          "|---|---|---|---|---|---|---|---|"]
    for did in per.index:
        r = per.loc[did]
        notes = drift.get(did, [])
        y = max(years[did], 1e-9)
        unmarked = int(((s["driver_id"] == did) & (s["level"] == "ALERT") & ~s["anomaly"]).sum())
        L.append(f"| {did} | {r['LEARNING']} | {r['NORMAL']} | {r['WATCH']} | {r['ALERT']} | "
                 f"{r['ALERT'] / y:.1f} | {unmarked / y:.1f} | {len(notes)}회 {', '.join(sorted(notes)[:3])} |")
    L += ["", f"시뮬레이션 기준 기대치: 표시 없는(평범한) 날 경고 1명당 연 약 {SIM['alerts_per_year']}회, "
              f"장기 변화 알림 연 약 {SIM['drift_per_year']}회(변화 없을 때)."]
    if has_labels:
        e = s[s["n_prev"] >= MIN_HISTORY]
        pos = e[e["anomaly"]]
        alerts = e[e["level"] == "ALERT"]
        L += ["", "## 4. 운전자가 표시한 날과 비교", "",
              f"- 표시한 날 {len(pos)}회 중 경고 {int((pos['level'] == 'ALERT').sum())}회, "
              f"주의 이상 {int(pos['level'].isin(['ALERT', 'WATCH']).sum())}회",
              f"- 경고 {len(alerts)}회 중 표시한 날 {int(alerts['anomaly'].sum())}회"]
    L += ["", "## 5. 다시 맞출 값 (제안)", ""]
    if sug["commutes_scored"] < 200:
        L.append(f"- 판정한 출퇴근이 {sug['commutes_scored']}회뿐이라 아직 다시 맞추기엔 적다(200회 이상 권장).")
    if var.get("groups", 0):
        L.append("- 특성 모집단 분산: " + ", ".join(f"{c} {var[f'var_{c}']:.4f}" for c in FEATURES)
                 + " (`PersonalBaseline.kt`의 POP_VAR_*)")
    if not np.isnan(sug["p_alert_for_target"]):
        L.append(f"- 경고가 1명당 연 {TARGET_ALERTS:.0f}회가 되는 p값 기준: {sug['p_alert_for_target']:.1e} "
                 f"(지금 {APP_P['alert']:.0e}, 운전자·연 {sug['driver_years']:.1f})")
    L.append("")
    return "\n".join(L)


def from_sim(dataset: Path, n_drivers: int, out: Path) -> list[Path]:
    """SUMO year dataset -> per-driver commutes.csv files in the app's format + labels.csv."""
    from personal_year import load_year
    data = load_year(dataset)
    t = data["trips"]
    ids = [d for d in data["plain"] if d in set(t["driver_id"])][:n_drivers]
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    cols = ["date", "window", "route", "km", "hard_brakes", "moderate_brakes", "accel_rms", "max_decel",
            "e_usual", "e_today"]
    for did in ids:
        g = t[t["driver_id"] == did].sort_values(["date", "window"])
        p = out / f"{did}.csv"
        g[cols].round(4).to_csv(p, index=False)
        paths.append(p)
    lab = t[t["driver_id"].isin(ids) & t["anomaly"]][["driver_id", "date", "window", "kind"]]
    lab.rename(columns={"kind": "note"}).to_csv(out / "labels.csv", index=False)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help="commutes.csv files pulled from cars")
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--from-sim", type=Path, help="SUMO year dataset folder: make app-format files to try this on")
    ap.add_argument("--drivers", type=int, default=5)
    args = ap.parse_args()
    files, labels = args.files, args.labels
    if args.from_sim:
        sim_out = HERE / "out" / "real_check_sim"
        files = from_sim(args.from_sim, args.drivers, sim_out)
        labels = sim_out / "labels.csv"
        print(f"SUMO commutes of {len(files)} drivers -> {sim_out}")
    if not files:
        ap.error("give commutes.csv files or --from-sim")
    t = attach_labels(load_commutes(files), labels)
    s = replay(t)
    md = report(t, s, drift_notes(t), variation(t), suggest(s, driver_years(t)), labels is not None)
    if args.out:
        args.out.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
