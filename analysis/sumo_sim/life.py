"""A year in the life of commuting drivers: the calendar and the changes planted on top of
each driver's base style, for the 1-year SUMO dataset (generate_year.py).

Things that should NOT raise an alarm about the driver
  * weather      rain / snow days change every vehicle's driving (incl. background traffic)
  * holidays     Korean public holidays: nobody commutes
  * departure    each driver has a usual departure time, +-~10 min day to day
  * daily form   small day-to-day variation of the driver's own style
  * vacation     a 1-2 week block plus a few single days off
  * route change a new workplace from some date on (new route, new congestion)
  * new vehicle  slightly smoother driving from some date on

Things that SHOULD
  * one-day anomalies  a single commute driven fatigued / hurried
  * long-term drift    fatigued / hurried with a shape:
        fast       full strength after 1-2 weeks, stays
        slow       ramps up over 2-3 months, stays
        transient  up over 2 weeks, holds 3-6 weeks, back to normal over 2 weeks

Pure functions (no SUMO), so they are unit-tested in test_life.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scenario import STATE_KINDS, style_variant

# Korean public holidays incl. substitute holidays (대체공휴일), 2026 and 1 Jan 2027.
HOLIDAYS = {
    "2026-01-01": "신정",
    "2026-02-16": "설날 연휴", "2026-02-17": "설날", "2026-02-18": "설날 연휴",
    "2026-03-02": "삼일절 대체공휴일",
    "2026-05-05": "어린이날",
    "2026-05-25": "부처님오신날 대체공휴일",
    "2026-06-03": "전국동시지방선거",
    "2026-08-17": "광복절 대체공휴일",
    "2026-09-24": "추석 연휴", "2026-09-25": "추석", "2026-09-26": "추석 연휴",
    "2026-10-05": "개천절 대체공휴일",
    "2026-10-09": "한글날",
    "2026-12-25": "성탄절",
    "2027-01-01": "신정",
}

# Approximate share of days with precipitation in Seoul by month (synthetic weather).
PRECIP_P = {1: 0.12, 2: 0.13, 3: 0.20, 4: 0.23, 5: 0.25, 6: 0.30,
            7: 0.50, 8: 0.45, 9: 0.28, 10: 0.18, 11: 0.23, 12: 0.15}
SNOW_SHARE = {12: 0.6, 1: 0.7, 2: 0.5, 3: 0.15}   # of precipitation days, falling as snow
PERSIST = 0.35                                      # extra chance of precipitation after a wet day

# How weather changes every vehicle's style (multiplicative / additive).
WEATHER_EFFECT = {
    "clear": {},
    # rain: worse perception outweighs the extra caution -> ~+25% hard brakes for everyone
    # (same windows, same traffic: clear 8.5 vs rain 10.6 per 100 passes). snow: much slower and
    # friction-limited braking -> ~-50% hard brakes. Checked in the 8-week pilot.
    "rain": {"treaction": ("+", 0.25), "sigmaerror": ("+", 0.08), "sigmagap": ("+", 0.08),
             "speedFactor": ("*", 0.94), "tau": ("*", 1.07), "decel": ("*", 0.90)},
    "snow": {"treaction": ("+", 0.35), "sigmaerror": ("+", 0.10), "sigmagap": ("+", 0.10),
             "speedFactor": ("*", 0.80), "tau": ("*", 1.25), "decel": ("*", 0.75), "accel": ("*", 0.80)},
}
WEATHER_TRAFFIC = {"clear": 1.0, "rain": 1.10, "snow": 0.90}   # background volume factor

JITTER_KEYS = ("accel", "decel", "tau", "speedFactor", "treaction")
DEPART_SD_S = 600.0          # day-to-day departure spread around the usual time


# ---------------------------------------------------------------- calendar

def workdays(dates: pd.DatetimeIndex) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in dates
            if d.dayofweek < 5 and d.strftime("%Y-%m-%d") not in HOLIDAYS]


def weather_calendar(dates: pd.DatetimeIndex, rng: np.random.Generator, mix: str = "seoul") -> dict[str, str]:
    """date -> clear / rain / snow. mix='uniform' gives each a third (pilot: effect sizes)."""
    out, wet = {}, False
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        if mix == "uniform":
            out[ds] = str(rng.choice(["clear", "rain", "snow"]))
            continue
        p = min(PRECIP_P[d.month] + (PERSIST if wet else 0.0), 0.9)
        wet = rng.random() < p
        out[ds] = ("snow" if rng.random() < SNOW_SHARE.get(d.month, 0.0) else "rain") if wet else "clear"
    return out


def apply_weather(p: dict, weather: str) -> dict:
    q = dict(p)
    for k, (op, v) in WEATHER_EFFECT[weather].items():
        q[k] = q[k] + v if op == "+" else q[k] * v
    return q


def jitter(p: dict, rng: np.random.Generator, sd: float = 0.04) -> dict:
    """Daily form: each of the main parameters scaled by 1 + N(0, sd), clipped to +-10%."""
    q = dict(p)
    for k in JITTER_KEYS:
        q[k] = q[k] * float(np.clip(1 + rng.normal(0, sd), 0.9, 1.1))
    return q


def new_vehicle(p: dict) -> dict:
    """A newer / more comfortable car: gentler acceleration, steadier following."""
    q = dict(p)
    q["accel"] *= 0.9
    q["sigmaerror"] *= 0.8
    q["sigmagap"] *= 0.8
    q["tau"] *= 1.05
    return q


# ---------------------------------------------------------------- per-driver plan

@dataclass
class Drift:
    kind: str          # fatigue / hurried
    shape: str         # fast / slow / transient
    start: int         # workday index
    ramp: int          # workdays to full strength
    hold: int = 0      # transient: workdays at full strength
    fall: int = 0      # transient: workdays back to normal

    def strength(self, i: int) -> float:
        if i < self.start:
            return 0.0
        k = i - self.start
        if k < self.ramp:
            return (k + 1) / self.ramp
        if self.shape != "transient":
            return 1.0
        k -= self.ramp
        if k < self.hold:
            return 1.0
        k -= self.hold
        return max(0.0, 1.0 - (k + 1) / self.fall) if k < self.fall else 0.0

    def end(self) -> int | None:
        return self.start + self.ramp + self.hold + self.fall if self.shape == "transient" else None


@dataclass
class DriverLife:
    driver_id: str
    depart_mean: dict                       # window -> usual departure (s of day)
    off_days: set = field(default_factory=set)        # workday indices without commuting
    vacation: tuple | None = None           # (first, last) workday index
    route_change: int | None = None         # workday index from which the new workplace applies
    vehicle_change: int | None = None
    drift: Drift | None = None
    anomalies: dict = field(default_factory=dict)     # (workday index, window) -> kind


def plan_lives(driver_ids: list[str], n_days: int, rng: np.random.Generator, windows: dict,
               drift_share: float = 0.3, route_share: float = 0.1, vehicle_share: float = 0.1,
               anomalies_per_year: float = 4.0, warmup: int = 10) -> dict[str, DriverLife]:
    """Plans every driver's year over n_days workdays. Drift, route change and vehicle change
    go to disjoint drivers so each can be evaluated on its own."""
    n = len(driver_ids)
    order = list(rng.permutation(driver_ids))
    n_drift, n_route, n_veh = round(drift_share * n), round(route_share * n), round(vehicle_share * n)
    drift_ids = set(order[:n_drift])
    route_ids = set(order[n_drift:n_drift + n_route])
    veh_ids = set(order[n_drift + n_route:n_drift + n_route + n_veh])
    scale = n_days / 245.0                    # a working year has ~245 workdays
    lives = {}
    for k, did in enumerate(driver_ids):
        depart = {}
        for w, (b, e) in windows.items():
            depart[w] = float(rng.uniform(b * 3600 + 900, e * 3600 - 1500))
        life = DriverLife(did, depart)

        # vacation: one block of 5-10 workdays, plus a few single days
        if n_days > 40:
            length = int(rng.integers(5, 11))
            first = int(rng.integers(warmup, n_days - length))
            life.vacation = (first, first + length - 1)
            life.off_days |= set(range(first, first + length))
        life.off_days |= set(int(x) for x in rng.choice(n_days, size=max(1, int(5 * scale)), replace=False))

        if did in drift_ids:
            shape = ("fast", "slow", "transient")[k % 3]
            kind = str(rng.choice(STATE_KINDS))
            lo, hi = max(warmup + 5, int(0.12 * n_days)), max(warmup + 6, int(0.7 * n_days))
            start = int(rng.integers(lo, hi))
            if shape == "fast":
                life.drift = Drift(kind, shape, start, int(rng.integers(5, 11)))
            elif shape == "slow":
                life.drift = Drift(kind, shape, start, max(5, int(rng.integers(40, 66) * min(1.0, scale))))
            else:
                life.drift = Drift(kind, shape, start, 10, int(rng.integers(15, 31)), 10)
        if did in route_ids:
            life.route_change = int(rng.integers(int(0.2 * n_days), int(0.8 * n_days)))
        if did in veh_ids:
            life.vehicle_change = int(rng.integers(int(0.2 * n_days), int(0.8 * n_days)))

        # one-day anomalies: on commutes without drift, after the warm-up
        n_anom = rng.poisson(anomalies_per_year * scale)
        free = [(i, w) for i in range(warmup, n_days) for w in windows
                if i not in life.off_days and (life.drift is None or life.drift.strength(i) == 0.0)]
        for j in rng.choice(len(free), size=min(n_anom, len(free)), replace=False):
            life.anomalies[free[j]] = str(rng.choice(STATE_KINDS))
        lives[did] = life
    return lives


def trip_style(base: dict, life: DriverLife, i: int, window: str, weather: str,
               rng: np.random.Generator) -> tuple[dict, str | None, float]:
    """Style of driver `life` on workday i: base -> vehicle -> daily form -> state -> weather.
    Returns (params, state kind or None, state strength)."""
    p = new_vehicle(base) if life.vehicle_change is not None and i >= life.vehicle_change else dict(base)
    p = jitter(p, rng)
    kind, s = None, 0.0
    if (i, window) in life.anomalies:
        kind, s = life.anomalies[(i, window)], 1.0
    elif life.drift is not None and life.drift.strength(i) > 0:
        kind, s = life.drift.kind, life.drift.strength(i)
    if kind:
        p = style_variant(p, kind, s)
    return apply_weather(p, weather), kind, s


def depart_time(life: DriverLife, window: str, windows: dict, rng: np.random.Generator) -> float:
    b, e = windows[window]
    return float(np.clip(life.depart_mean[window] + rng.normal(0, DEPART_SD_S), b * 3600, e * 3600 - 600))
