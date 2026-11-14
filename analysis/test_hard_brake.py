"""Mirrors app/src/test/.../HardBrakeDetectorTest.kt so the Python port stays equivalent."""
from hard_brake import HardBrakeDetector

HZ = 10


def run(profile, det=None):
    det = det or HardBrakeDetector()
    return [e for i, v in enumerate(profile) if (e := det.on_sample(i / HZ, v))]


def cruise(v, sec):
    return [v] * int(sec * HZ)


def ramp(frm, to, accel):
    out, v, dv = [], frm, accel / HZ
    while (v + dv > to) if dv < 0 else (v + dv < to):
        v += dv
        out.append(v)
    return out + [to]


def test_hard_stop_from_60_is_one_event():
    v60 = 60 / 3.6
    ev = run(cruise(v60, 3) + ramp(v60, 0, -6) + cruise(0, 2))
    assert len(ev) == 1
    assert 5 <= ev[0].peak_decel <= 7
    assert ev[0].start_speed > 15 and ev[0].end_speed < 1


def test_gentle_stop_is_not_an_event():
    v60 = 60 / 3.6
    assert run(cruise(v60, 3) + ramp(v60, 0, -1.5) + cruise(0, 2)) == []


def test_hard_slowdown_without_stopping():
    v80, v40 = 80 / 3.6, 40 / 3.6
    ev = run(cruise(v80, 3) + ramp(v80, v40, -4.5) + cruise(v40, 3))
    assert len(ev) == 1 and abs(ev[0].end_speed - v40) < 1


def test_crawl_speed_ignored():
    v8 = 8 / 3.6
    assert run(cruise(v8, 2) + ramp(v8, 0, -5) + cruise(0, 2)) == []


def test_two_separate_events():
    v50, v20 = 50 / 3.6, 20 / 3.6
    p = cruise(v50, 2) + ramp(v50, v20, -5) + cruise(v20, 2) + ramp(v20, v50, 2) + cruise(v50, 2) + ramp(v50, 0, -5) + cruise(0, 1)
    assert len(run(p)) == 2


def test_works_at_5hz():
    # The SUMO generator samples at 5 Hz (0.2 s steps).
    det = HardBrakeDetector()
    v60 = 60 / 3.6
    prof = cruise(v60, 3) + ramp(v60, 0, -6) + cruise(0, 2)
    ev = [e for i, v in enumerate(prof[::2]) if (e := det.on_sample(i * 0.2, v))]
    assert len(ev) == 1
