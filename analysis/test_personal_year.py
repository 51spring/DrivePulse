"""Building blocks of the 1-year evaluation (personal_year.py) and the history options of
personal.score_trips."""
import numpy as np
import pandas as pd

from personal import _past_sum
from personal_year import REF_DAYS, _ref_sum, alarms


def test_past_sum_histories():
    df = pd.DataFrame({"k": ["a"] * 6, "x": [1.0, 2, 3, 4, 5, 6]})
    g = df.groupby("k")
    assert _past_sum(g, "x", None).tolist() == [0, 1, 3, 6, 10, 15]
    assert _past_sum(g, "x", ("window", 2)).tolist() == [0, 1, 3, 5, 7, 9]
    assert _past_sum(g, "x", ("fixed", 2)).tolist() == [0, 1, 3, 3, 3, 3]


def test_reference_frozen_vs_lagged():
    x = np.ones(200)
    s, n = _ref_sum(x, None)
    assert (n == REF_DAYS).all() and (s == REF_DAYS).all()
    s, n = _ref_sum(x, (20, 60))
    assert n[30] == REF_DAYS                   # early on: same as frozen
    assert n[150] == 60 and s[150] == 60       # later: the 60 days ending 20 days ago


def _daily(z):
    n = len(z)
    return pd.DataFrame({"driver_id": "a", "day_idx": np.arange(n), "z_m": z, "hard_per_km": 0.0})


def test_cusum_needs_a_sustained_shift_and_restarts():
    one_bad_day = np.zeros(100)
    one_bad_day[50] = 3.0                      # a capped single-day anomaly
    assert alarms(_daily(one_bad_day), "m", h=4.0)["a"] == []
    shift = np.zeros(100)
    shift[40:] = 1.5                           # sustained: +1 per day after the slack
    hits = alarms(_daily(shift), "m", h=4.0)["a"]
    assert hits[0] == 44 and len(hits) > 5     # alarms again after each restart
