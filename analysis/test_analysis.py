"""Run: python -m pytest -q   (from the analysis/ folder)"""
import numpy as np
import pandas as pd

from baseline import add_context, benjamini_hochberg, detect_driver_anomalies, detect_segment_anomalies
from ingest_app import ingest


def _flat_traffic(n_days=28, per_cell=40, rate=0.02, seed=0, spike=None):
    """Two segments, weekday am_peak only; optional spike=(segment, date, mult)."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.bdate_range("2026-09-01", periods=n_days).strftime("%Y-%m-%d")
    for seg in ("A", "B"):
        for dt in dates:
            mult = spike[2] if spike and (seg, dt) == spike[:2] else 1.0
            for i in range(per_cell):
                rows.append(("t", f"d{i}", seg, dt, 8, rng.poisson(rate * mult)))
    return add_context(pd.DataFrame(rows, columns=["trip_id", "driver_id", "segment_id", "date", "hour", "hard_brakes"]))


def test_bh_matches_reference():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    np.testing.assert_allclose(benjamini_hochberg(p), [0.04, 0.0533333, 0.0533333, 0.20], rtol=1e-5)


def test_no_flags_on_flat_traffic():
    cells = detect_segment_anomalies(_flat_traffic())
    assert cells["flagged"].sum() == 0


def test_spike_is_flagged():
    df = _flat_traffic(spike=("B", "2026-09-10", 10.0))
    cells = detect_segment_anomalies(df)
    flagged = cells[cells["flagged"]]
    assert list(zip(flagged["segment_id"], flagged["date"])) == [("B", "2026-09-10")]


def test_short_spike_found_in_30min_slots_but_diluted_in_band():
    # 8 weeks of weekday mornings, 7:00-9:59, 100 passes per 30-min slot, 2% base rate.
    # One day has a 30-min spike (x8, ~16 vs ~2 expected) at 08:00-08:29 on segment B.
    rng = np.random.default_rng(3)
    rows = []
    dates = pd.bdate_range("2026-09-01", periods=40).strftime("%Y-%m-%d")
    for dt in dates:
        for seg in ("A", "B"):
            for slot in range(6):
                hour, minute = 7 + slot // 2, (slot % 2) * 30
                mult = 8.0 if (seg, dt, hour, minute) == ("B", dates[20], 8, 0) else 1.0
                for i in range(100):
                    rows.append(("t", f"d{i}", seg, dt, hour, minute + i % 30, rng.poisson(0.02 * mult)))
    raw = pd.DataFrame(rows, columns=["trip_id", "driver_id", "segment_id", "date", "hour", "minute", "hard_brakes"])

    slot = detect_segment_anomalies(add_context(raw, slot_minutes=30))
    top = slot.iloc[0]
    assert (top["segment_id"], top["date"], top["bucket"]) == ("B", dates[20], "weekday_08:00")
    assert top["flagged"]

    band = detect_segment_anomalies(add_context(raw))
    b = band[(band["segment_id"] == "B") & (band["date"] == dates[20])].iloc[0]
    assert b["rate_ratio"] < top["rate_ratio"]  # the same spike looks weaker over the whole band


def test_driver_sir_uses_context():
    # Segment A is 4x riskier than B. Driver "a" only drives A, "b" only B; both normal.
    rng = np.random.default_rng(1)
    rows = []
    for dt in pd.bdate_range("2026-09-01", periods=40).strftime("%Y-%m-%d"):
        for i in range(30):
            rows.append(("t", f"a{i}", "A", dt, 8, rng.poisson(0.08)))
            rows.append(("t", f"b{i}", "B", dt, 8, rng.poisson(0.02)))
    df = add_context(pd.DataFrame(rows, columns=["trip_id", "driver_id", "segment_id", "date", "hour", "hard_brakes"]))
    drivers = detect_driver_anomalies(df, detect_segment_anomalies(df))
    a = drivers[drivers["driver_id"].str.startswith("a")]
    b = drivers[drivers["driver_id"].str.startswith("b")]
    assert a["naive_rate_per100"].mean() > 3 * b["naive_rate_per100"].mean()  # absolute score blames A drivers
    assert abs(a["sir"].mean() - b["sir"].mean()) < 0.3                       # context score does not
    assert drivers["flagged"].sum() == 0


def test_ingest_app_logs(tmp_path):
    t0 = pd.Timestamp("2026-11-05 08:00:00")
    rows = []
    for i in range(60):  # 6 s at 10 Hz: 3 s in cell1, 3 s in cell2
        gh = "wydm9qa" if i < 30 else "wydm9qb"
        rows.append((i * 100, (t0 + pd.Timedelta(milliseconds=100 * i)).isoformat(), 40.0, "", "", "", "", "", "", gh))
    pd.DataFrame(rows, columns=["elapsed_ms", "wall_time", "speed_kmh", "accel_mps2", "brake", "gear",
                                "steering_deg", "lat", "lon", "geohash"]).to_csv(tmp_path / "s1_samples.csv", index=False)
    pd.DataFrame([{"session_id": "s1", "start_time": (t0 + pd.Timedelta(seconds=4)).isoformat(), "geohash": "wydm9qb"}]) \
        .to_csv(tmp_path / "events.csv", index=False)

    trav = ingest(tmp_path, "car01")
    assert list(trav["segment_id"]) == ["wydm9qa", "wydm9qb"]
    assert list(trav["hard_brakes"]) == [0, 1]
    assert set(trav["hour"]) == {8}
