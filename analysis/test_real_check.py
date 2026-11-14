"""real_check.py on a small hand-made commute file in the app's format."""
import numpy as np
import pandas as pd

from real_check import attach_labels, driver_years, load_commutes, replay, report, suggest, variation


def _commutes(tmp_path):
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2026-03-02", periods=60).strftime("%Y-%m-%d")
    rows = []
    for i, d in enumerate(days):
        tired = d == days[40]
        rows.append({"date": d, "window": "am", "route": "A>B", "km": 6.0,
                     "hard_brakes": 6 if tired else int(rng.poisson(0.3)),
                     "moderate_brakes": 14 if tired else int(rng.poisson(4)),
                     "accel_rms": 0.9 if tired else 0.6 + rng.normal(0, 0.03),
                     "max_decel": 5.5 if tired else 2.8 + rng.normal(0, 0.3),
                     "e_usual": 1.2, "e_today": "" if i % 10 == 0 else 1.2})   # some trips without server data
    f = tmp_path / "car01.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    lab = tmp_path / "labels.csv"
    pd.DataFrame([{"driver_id": "car01", "date": days[40], "window": "am", "note": "피곤"}]).to_csv(lab, index=False)
    return f, lab


def test_replay_flags_the_marked_day_and_reports(tmp_path):
    f, lab = _commutes(tmp_path)
    t = attach_labels(load_commutes([f]), lab)
    assert t["e_today"].notna().all()                       # server gaps filled like the app
    s = replay(t)
    assert (s["level"] == "LEARNING").sum() == 3
    assert s.loc[s["anomaly"], "level"].item() == "ALERT"
    assert (s.loc[~s["anomaly"], "level"] == "ALERT").sum() <= 1
    var = variation(t)
    assert var["groups"] == 1 and 0 < var["accel_cv"] < 0.2
    md = report(t, s, {}, var, suggest(s, driver_years(t)), True)
    assert "표시한 날 1회 중 경고 1회" in md
