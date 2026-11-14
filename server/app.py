"""DrivePulse alert server (development / demo).

Cars upload traversals (one row per road segment passed, with its hard-brake count).
The server runs the same detectors as the analysis (analysis/baseline.py) and serves
segments whose hard braking is spiking right now compared with their usual level at
that time.

    python server/app.py seed analysis/out/exp/gangnam_s3/traversals.csv   # load history
    python server/app.py demo-today --segment 919983793#3 --start 07:10     # demo spike today
    python server/app.py serve --port 8000

API
    POST /api/traversals    {"traversals": [{"segment_id",
                             "date", "hour", "minute", "hard_brakes"}, ...]}
    POST /api/trip_context  {"traversals": [{"segment_id", "date", "hour", "minute"}, ...]}
                            -> {"e_usual", "e_today"}: expected hard brakes on those passes usually /
                               given other cars today (road-condition adjustment for the car's
                               personal baseline; the car keeps its own history)
    GET  /api/alerts?date=YYYY-MM-DD&hour=H&minute=M   -> {"alerts": [...], "computed_at": ...}
    GET  /api/health

Privacy: uploads are anonymous. Each row is one segment pass (segment, date, time, hard
brakes) with no driver or trip ID, so passes can't be linked into a car's route. The car's
personal commute history and its reports never leave the car.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from baseline import add_context, detect_segment_anomalies, detect_window_spikes  # noqa: E402

DB = Path(__file__).resolve().parent / "data" / "drivepulse.db"
SEGMENTS = ROOT / "app" / "src" / "main" / "assets" / "segments_gangnam.csv"
KST = ZoneInfo("Asia/Seoul")
COLUMNS = ["trip_id", "driver_id", "segment_id", "date", "hour", "minute", "hard_brakes"]
RECOMPUTE_EVERY_S = 60


def connect() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB, check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS traversals (
        trip_id TEXT, driver_id TEXT, segment_id TEXT, date TEXT, hour INTEGER, minute INTEGER,
        hard_brakes INTEGER, source TEXT DEFAULT 'app', received_at TEXT)""")
    con.execute("CREATE INDEX IF NOT EXISTS ix_date ON traversals(date)")
    return con


class Detector:
    """Recomputes detection in the background; requests read the latest result."""

    def __init__(self, con: sqlite3.Connection):
        self.con = con
        self.lock = threading.Lock()
        self.result = None          # (band_cells, scan, computed_at, n_rows)
        self.dirty = True
        self.last_run = 0.0
        self.names = {}
        if SEGMENTS.exists():
            seg = pd.read_csv(SEGMENTS, usecols=["edge_id", "name"])
            self.names = dict(zip(seg["edge_id"], seg["name"].fillna("")))
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            if self.dirty and time.time() - self.last_run >= RECOMPUTE_EVERY_S or self.result is None:
                self._compute()
            time.sleep(2)

    def _compute(self):
        self.dirty = False
        self.last_run = time.time()
        df = pd.read_sql_query(f"SELECT {', '.join(COLUMNS)} FROM traversals", self.con)
        if df.empty:
            with self.lock:
                self.result = (pd.DataFrame(), pd.DataFrame(), datetime.now(KST).isoformat(), 0)
            return
        ctx = add_context(df)
        cells = detect_segment_anomalies(ctx)
        scan = detect_window_spikes(ctx, cells)
        # road-condition lookups for the in-car personal baseline
        cell_counts = {(s, b, d): (int(c), int(e)) for s, b, d, c, e in
                       cells[["segment_id", "bucket", "date", "count", "exposure"]].itertuples(index=False)}
        sb = cells.groupby(["segment_id", "bucket"])[["count", "exposure"]].sum()
        sb_counts = {k: (int(r["count"]), int(r["exposure"])) for k, r in sb.iterrows()}
        global_rate = cells["count"].sum() / max(cells["exposure"].sum(), 1)
        with self.lock:
            self.result = (cells, scan, datetime.now(KST).isoformat(timespec="seconds"), len(df))
            self.road = (cell_counts, sb_counts, global_rate)
        print(f"[detector] {len(df):,} traversals, {int(scan['flagged'].sum())} flagged windows", flush=True)

    def trip_context(self, traversals: list[dict], k: float = 20.0) -> dict:
        """Expected hard brakes of a trip's segment passes: usually, and given other cars today.
        Used by the car to separate "my driving changed" from "the road was bad today"."""
        with self.lock:
            road = getattr(self, "road", None)
        if road is None or not traversals:
            return {"e_usual": None, "e_today": None, "matched": 0}
        cell_counts, sb_counts, global_rate = road
        rows = pd.DataFrame(traversals)
        rows = rows.assign(trip_id="", driver_id="", hard_brakes=0)
        rows["bucket"] = add_context(rows)["bucket"]
        e_usual = e_today = 0.0
        matched = 0
        for r in rows.itertuples():
            y_all, n_all = sb_counts.get((r.segment_id, r.bucket), (0, 0))
            y_day, n_day = cell_counts.get((r.segment_id, r.bucket, r.date), (0, 0))
            y_hist, n_hist = y_all - y_day, n_all - n_day
            usual = (y_hist + k * global_rate) / (n_hist + k)
            today = (y_day + k * usual) / (n_day + k)
            e_usual += usual
            e_today += today
            matched += n_hist > 0
        return {"e_usual": round(e_usual, 4), "e_today": round(e_today, 4), "matched": int(matched)}

    def alerts(self, date: str, hour: int, minute: int) -> dict:
        with self.lock:
            if self.result is None:
                return {"alerts": [], "computed_at": None, "ready": False}
            cells, scan, computed_at, n = self.result
        if scan.empty:
            return {"alerts": [], "computed_at": computed_at, "ready": True}
        bucket = add_context(pd.DataFrame([{c: 0 for c in COLUMNS} | {"date": date, "hour": hour}]))["bucket"][0]
        now_min = hour * 60 + minute
        s = scan[(scan["date"] == date) & (scan["bucket"] == bucket) & scan["flagged"]].copy()
        start = s["window_start"].str.slice(0, 2).astype(int) * 60 + s["window_start"].str.slice(3, 5).astype(int)
        s = s[(start <= now_min) & (now_min <= start + s["window_minutes"] + 30)]  # recent spikes only
        alerts = [{
            "segment_id": r.segment_id,
            "name": self.names.get(r.segment_id, ""),
            "window_start": r.window_start,
            "window_minutes": int(r.window_minutes),
            "observed": int(r.w_count),
            "expected": round(float(r.w_expected), 1),
            "ratio": round(float(r.w_ratio), 1),
        } for r in s.sort_values("p_value").itertuples()]
        return {"alerts": alerts, "computed_at": computed_at, "ready": True, "traversals": n}


def create_app() -> Flask:
    con = connect()
    detector = Detector(con)
    app = Flask(__name__)

    @app.get("/api/health")
    def health():
        n = con.execute("SELECT COUNT(*) FROM traversals").fetchone()[0]
        return jsonify(ok=True, traversals=n)

    @app.post("/api/traversals")
    def upload():
        body = request.get_json(force=True, silent=True) or {}
        driver = ""  # uploads are anonymous: driver/trip IDs are not stored even if sent
        rows = []
        for t in body.get("traversals", [])[:5000]:
            try:
                rows.append(("", driver, str(t["segment_id"])[:80], str(t["date"])[:10],
                             int(t["hour"]), int(t["minute"]), int(t["hard_brakes"]), "app",
                             datetime.now(KST).isoformat(timespec="seconds")))
            except (KeyError, TypeError, ValueError):
                return jsonify(error=f"bad traversal: {t}"), 400
        con.executemany("INSERT INTO traversals VALUES (?,?,?,?,?,?,?,?,?)", rows)
        con.commit()
        detector.dirty = True
        return jsonify(stored=len(rows))

    @app.post("/api/trip_context")
    def trip_context():
        body = request.get_json(force=True, silent=True) or {}
        trav = [{"segment_id": str(t["segment_id"]), "date": str(t["date"])[:10], "hour": int(t["hour"]),
                 "minute": int(t.get("minute", 0))} for t in body.get("traversals", [])[:2000]
                if "segment_id" in t and "date" in t and "hour" in t]
        return jsonify(detector.trip_context(trav))

    @app.get("/api/alerts")
    def alerts():
        now = datetime.now(KST)
        date = request.args.get("date", now.strftime("%Y-%m-%d"))
        hour = int(request.args.get("hour", now.hour))
        minute = int(request.args.get("minute", now.minute))
        return jsonify(detector.alerts(date, hour, minute))

    return app


def seed(csv_file: Path):
    con = connect()
    df = pd.read_csv(csv_file, usecols=COLUMNS)
    df["source"] = "sim_history"
    df["received_at"] = ""
    df.to_sql("traversals", con, if_exists="append", index=False, chunksize=50_000)
    print(f"seeded {len(df):,} traversals from {csv_file}")


def demo_today(segment: str, start: str, date: str | None, minutes: int, share: float):
    """DEMO ONLY: copies one historical weekday morning onto `date` and adds hard brakes on
    `segment` for `minutes` from `start`, so the alert path can be shown end to end."""
    con = connect()
    date = date or datetime.now(KST).strftime("%Y-%m-%d")
    hist = pd.read_sql_query(f"SELECT {', '.join(COLUMNS)} FROM traversals WHERE source='sim_history'", con)
    weekday = pd.to_datetime(hist["date"]).dt.dayofweek < 5
    source_day = sorted(hist.loc[weekday, "date"].unique())[0]
    day = hist[hist["date"] == source_day].copy()
    day["date"] = date
    t = day["hour"] * 60 + day["minute"]
    h, m = map(int, start.split(":"))
    hit = (day["segment_id"] == segment) & (t >= h * 60 + m) & (t < h * 60 + m + minutes)
    day.loc[hit & (pd.Series(range(len(day)), index=day.index) % round(1 / share) == 0), "hard_brakes"] += 1
    con.execute("DELETE FROM traversals WHERE date = ? AND source = 'demo'", (date,))
    day["source"] = "demo"
    day["received_at"] = ""
    day.to_sql("traversals", con, if_exists="append", index=False)
    con.commit()
    print(f"demo: copied {source_day} -> {date} ({len(day):,} traversals), "
          f"{int(hit.sum())} passes on {segment} {start}+{minutes}min get extra hard brakes")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s = sub.add_parser("seed")
    s.add_argument("csv", type=Path)
    s = sub.add_parser("demo-today")
    s.add_argument("--segment", required=True)
    s.add_argument("--start", default="07:10")
    s.add_argument("--date")
    s.add_argument("--minutes", type=int, default=20)
    s.add_argument("--share", type=float, default=0.5, help="share of passes that get an extra hard brake")
    args = ap.parse_args()
    if args.cmd == "serve":
        create_app().run(host=args.host, port=args.port, threaded=True)
    elif args.cmd == "seed":
        seed(args.csv)
    else:
        demo_today(args.segment, args.start, args.date, args.minutes, args.share)


if __name__ == "__main__":
    main()
