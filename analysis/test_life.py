"""Planning of the 1-year dataset (sumo_sim/life.py): calendar, drift shapes, life events."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "sumo_sim"))
from life import (Drift, apply_weather, depart_time, jitter, plan_lives, trip_style,  # noqa: E402
                  weather_calendar, workdays)

WINDOWS = {"am": (7, 9), "pm": (17, 19)}
BASE = dict(accel=2.6, decel=2.2, tau=1.2, speedFactor=1.0, minGap=2.5, treaction=0.5,
            sigmaerror=0.1, sigmagap=0.1, sigmaleader=0.05, lcAssertive=1.0)


def test_workdays_skip_weekends_and_holidays():
    wd = workdays(pd.date_range("2026-02-16", "2026-02-22"))   # Seollal Mon-Wed
    assert wd == ["2026-02-19", "2026-02-20"]
    year = workdays(pd.date_range("2026-01-05", periods=364))
    assert 240 <= len(year) <= 250


def test_seoul_weather_is_seasonal():
    dates = pd.date_range("2026-01-01", "2026-12-31")
    w = pd.Series(weather_calendar(dates, np.random.default_rng(0)))
    month = pd.to_datetime(w.index).month
    assert (w[month == 7] != "clear").mean() > (w[month == 10] != "clear").mean()
    assert (w[(month >= 6) & (month <= 9)] == "snow").sum() == 0
    assert (w[month <= 2] == "snow").sum() > 0


def test_weather_makes_driving_more_cautious_and_less_precise():
    rain, snow = apply_weather(BASE, "rain"), apply_weather(BASE, "snow")
    assert snow["treaction"] > rain["treaction"] > BASE["treaction"]
    assert snow["speedFactor"] < rain["speedFactor"] < BASE["speedFactor"]
    assert apply_weather(BASE, "clear") == BASE


def test_jitter_stays_within_ten_percent():
    rng = np.random.default_rng(1)
    for _ in range(200):
        j = jitter(BASE, rng, sd=0.2)
        assert all(0.9 * BASE[k] - 1e-9 <= j[k] <= 1.1 * BASE[k] + 1e-9 for k in BASE)


def test_drift_shapes():
    fast = Drift("fatigue", "fast", start=20, ramp=5)
    assert fast.strength(19) == 0 and fast.strength(22) == 0.6 and fast.strength(200) == 1.0
    slow = Drift("hurried", "slow", start=20, ramp=50)
    assert 0 < slow.strength(40) < 1 and slow.strength(80) == 1.0
    tr = Drift("fatigue", "transient", start=20, ramp=10, hold=20, fall=10)
    assert tr.strength(35) == 1.0 and 0 < tr.strength(55) < 1 and tr.strength(60) == 0.0
    assert tr.end() == 60


def test_plan_lives_assigns_disjoint_changes():
    ids = [f"drv_{i:04d}" for i in range(100)]
    lives = plan_lives(ids, 245, np.random.default_rng(3), WINDOWS)
    drift = {d for d, l in lives.items() if l.drift}
    route = {d for d, l in lives.items() if l.route_change is not None}
    veh = {d for d, l in lives.items() if l.vehicle_change is not None}
    assert (len(drift), len(route), len(veh)) == (30, 10, 10)
    assert not (drift & route or drift & veh or route & veh)
    assert {l.drift.shape for l in lives.values() if l.drift} == {"fast", "slow", "transient"}
    n_anom = sum(len(l.anomalies) for l in lives.values())
    assert 300 < n_anom < 500
    for l in lives.values():   # anomalies never on a day off or during a drift
        for i, _ in l.anomalies:
            assert i not in l.off_days and i >= 10
            assert l.drift is None or l.drift.strength(i) == 0


def test_trip_style_layers_state_and_weather():
    lives = plan_lives(["a"], 245, np.random.default_rng(0), WINDOWS, drift_share=0, anomalies_per_year=0)
    life = lives["a"]
    life.anomalies[(50, "am")] = "fatigue"
    rng = np.random.default_rng(0)
    tired, kind, s = trip_style(BASE, life, 50, "am", "clear", rng)
    plain, kind0, _ = trip_style(BASE, life, 50, "pm", "clear", rng)
    assert (kind, s, kind0) == ("fatigue", 1.0, None)
    assert tired["treaction"] > plain["treaction"] + 0.5
    snowy, _, _ = trip_style(BASE, life, 51, "am", "snow", rng)
    assert snowy["speedFactor"] < 0.9


def test_departure_stays_in_window_and_near_usual_time():
    lives = plan_lives(["a"], 245, np.random.default_rng(0), WINDOWS)
    rng = np.random.default_rng(0)
    deps = np.array([depart_time(lives["a"], "am", WINDOWS, rng) for _ in range(500)])
    assert deps.min() >= 7 * 3600 and deps.max() <= 9 * 3600 - 600
    assert abs(np.median(deps) - lives["a"].depart_mean["am"]) < 120
