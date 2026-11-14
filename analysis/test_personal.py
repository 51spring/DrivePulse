"""Personal-baseline detectors on a small synthetic fleet with known answers."""
import numpy as np
import pandas as pd

from personal import (CALIB_DAYS, daily_series, drift_statistics, evaluate_day, evaluate_drift, score_trips,
                      trip_expectations)


def _fleet(seed=0):
    """40 drivers commute (am) over a 6-segment route for 30 weekdays.
    Segment S0 is congested (high hard-brake rate). Half the drivers use it.
    drv_00..03: base 'aggressive' (3x). One-day anomaly: drv_10 on day 20 (6x).
    Drift: drv_20 from day 15 on (3x). Road event: day 25 everyone on S0 brakes 4x."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-09-01", periods=30).strftime("%Y-%m-%d")
    seg_rate = {"S0": 0.25, "S1": 0.03, "S2": 0.03, "S3": 0.03, "S4": 0.03, "S5": 0.03}
    trav, trips = [], []
    for di, date in enumerate(dates):
        for i in range(40):
            did = f"drv_{i:02d}"
            segs = ["S0", "S1", "S2"] if i % 2 == 0 else ["S3", "S4", "S5"]
            mult = 3.0 if i < 4 else 1.0
            if did == "drv_10" and di == 20:
                mult *= 6
            if did == "drv_20" and di >= 15:
                mult *= 3
            tid = f"{did}#{date}am"
            hb_total = 0
            for s in segs:
                road = 4.0 if (s == "S0" and di == 25) else 1.0
                hb = rng.poisson(seg_rate[s] * mult * road)
                hb_total += hb
                trav.append((tid, did, s, date, 8, 0, hb))
            trips.append({"trip_id": tid, "driver_id": did, "date": date, "window": "am", "trip_kind": "commute",
                          "distance_m": 3000.0, "hard_brakes": hb_total,
                          "moderate_brakes": hb_total + rng.poisson(2 * mult),
                          "accel_rms": 0.8 * mult**0.3 + rng.normal(0, 0.03),
                          "max_decel": 2.5 + 0.5 * hb_total + rng.normal(0, 0.1),
                          "label": "aggressive" if i < 4 else "normal"})
    trav = pd.DataFrame(trav, columns=["trip_id", "driver_id", "segment_id", "date", "hour", "minute", "hard_brakes"])
    t = pd.DataFrame(trips).merge(trip_expectations(trav), on="trip_id")
    t["km"] = t["distance_m"] / 1000
    t["moderate_per_km"] = t["moderate_brakes"] / t["km"]
    t["hard_per_km"] = t["hard_brakes"] / t["km"]
    return t.sort_values(["driver_id", "window", "date"]).reset_index(drop=True), dates


def test_road_event_is_not_blamed_on_drivers():
    t, dates = _fleet()
    s = score_trips(t)
    day25 = s[(s["date"] == dates[25]) & (s["driver_id"].str[-2:].astype(int) % 2 == 0)]
    # Everyone on S0 braked more that day: today's road expectation absorbs it.
    assert day25["e_today"].mean() > 2 * day25["e_usual"].mean()
    assert day25["s_personal_road"].mean() < day25["s_personal"].mean()


def test_one_day_anomaly_ranks_first_with_personal_baseline():
    t, dates = _fleet()
    s = score_trips(t)
    truth = pd.DataFrame([{"driver_id": "drv_10", "date": dates[20], "window": "am", "kind": "fatigue"}])
    res, e = evaluate_day(s, truth, drift_ids={"drv_20"})
    assert res["n_anomalies"] == 1
    assert res["p@k_personal_road_f"] == 1.0
    # A permanently aggressive driver should not dominate the personal methods
    top_personal = e.nlargest(5, "s_personal_road")["driver_id"]
    assert not top_personal.isin(["drv_00", "drv_01", "drv_02", "drv_03"]).all()


def test_drift_detected_after_start():
    t, dates = _fleet()
    daily = drift_statistics(daily_series(t))
    truth = pd.DataFrame([{"driver_id": "drv_20", "kind": "hurried", "start_date": dates[15], "ramp_days": 1}])
    res = evaluate_drift(daily, truth, false_alarm=0.10)
    # drv_20's route has very few hard brakes (~0.1 per trip), so evidence builds up slowly
    assert res["detect_personal_road"] == 1.0
    assert res["delay_personal_road"] <= 10
    assert res["detect_personal_road_f"] == 1.0
    assert CALIB_DAYS < 15
