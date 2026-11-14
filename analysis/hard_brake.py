"""Python port of the app's HardBrakeDetector (app/.../drive/HardBrakeDetector.kt).

Same algorithm and defaults, so simulated data goes through the detector the car uses:
least-squares deceleration over the last 0.5 s, start at >= 3.0 m/s^2 while moving
>= 10 km/h, end below 1.5 m/s^2 or at standstill, minimum 0.3 s, re-arm after each event.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class HardBrakeEvent:
    start: float          # seconds
    end: float
    start_speed: float    # m/s
    end_speed: float
    peak_decel: float     # m/s^2, positive

    @property
    def duration(self) -> float:
        return self.end - self.start


class HardBrakeDetector:
    __slots__ = ("start_decel", "end_decel", "min_speed", "window", "min_duration", "max_gap",
                 "_t", "_v", "_active", "_armed", "last_accel")

    def __init__(self, start_decel=3.0, end_decel=1.5, min_speed=10 / 3.6,
                 window=0.5, min_duration=0.3, max_gap=2.0):
        self.start_decel = start_decel
        self.end_decel = end_decel
        self.min_speed = min_speed
        self.window = window
        self.min_duration = min_duration
        self.max_gap = max_gap
        self._t: deque = deque()
        self._v: deque = deque()
        self._active = None   # [start_t, start_speed, peak]
        self._armed = True
        self.last_accel = None

    def reset(self):
        self._t.clear()
        self._v.clear()
        self._active = None
        self._armed = True
        self.last_accel = None

    def on_sample(self, t: float, v: float):
        """Feeds one sample; returns a HardBrakeEvent when one completes, else None."""
        ts, vs = self._t, self._v
        if ts:
            if t <= ts[-1]:
                return None
            if t - ts[-1] > self.max_gap:
                self.reset()
        ts.append(t)
        vs.append(v)
        while len(ts) > 2 and t - ts[0] > self.window:
            ts.popleft()
            vs.popleft()
        n = len(ts)
        if n < 2:
            return None

        mt = sum(ts) / n
        mv = sum(vs) / n
        num = den = 0.0
        for ti, vi in zip(ts, vs):
            dt = ti - mt
            num += dt * (vi - mv)
            den += dt * dt
        if den <= 0:
            return None
        accel = num / den
        self.last_accel = accel
        decel = -accel

        a = self._active
        if a is None:
            if not self._armed:
                if decel < self.end_decel:
                    self._armed = True
                return None
            if decel >= self.start_decel and max(vs) >= self.min_speed and v >= 0.5:
                self._active = [ts[0], max(vs), decel]
            return None

        if decel > a[2]:
            a[2] = decel
        if decel >= self.end_decel and v >= 0.5:
            return None
        self._active = None
        self._armed = False
        if t - a[0] < self.min_duration:
            return None
        return HardBrakeEvent(a[0], t, a[1], v, a[2])
