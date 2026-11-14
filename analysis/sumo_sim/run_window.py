"""Runs one SUMO simulation window (one date x one time window) with libsumo and turns
every vehicle's 5 Hz speed trace into traversals + hard-brake events.

Vehicles' speeds go through the same HardBrakeDetector the car app uses
(analysis/hard_brake.py). Each traversal = one vehicle on one edge; time spent on
junction-internal edges counts toward the approach edge (braking for a signal belongs to
the road leading into it).

Incidents ("debris" mid-edge): while active, each vehicle reaching the middle of the
incident edge at > 6 m/s stops abruptly (5.5 m/s^2 to 1 m/s) with probability p, which
also makes followers brake hard.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hard_brake import HardBrakeDetector  # noqa: E402

STEP = 0.2
VAR_SPEED = 0x40
VAR_ROAD_ID = 0x50
VAR_LANE_INDEX = 0x52
INCIDENT_DECEL = 5.5  # m/s^2, abrupt stop for an obstacle
SPEED_MODE_DEFAULT = 31
SPEED_MODE_NO_DECEL_LIMIT = 31 & ~4  # bit 2 = regard maximum deceleration


def run_window(task: dict) -> dict:
    """task keys: net, vtypes, date, begin, end, trips (list of dicts), incidents, seed;
    optional count_lanes: also count each named driver's lane changes (trips: lane_changes)."""
    import libsumo as traci

    rng = np.random.default_rng(task["seed"])
    traci.start([
        "sumo", "-n", task["net"], "-a", task["vtypes"],
        "--begin", str(task["begin"]), "--end", str(task["end"]),
        "--step-length", str(STEP), "--seed", str(task["seed"]),
        "--no-step-log", "true", "--no-warnings", "true",
        "--time-to-teleport", "180", "--ignore-route-errors", "true",
        "--default.action-step-length", str(STEP),
    ])
    try:
        return _simulate(traci, task, rng)
    finally:
        traci.close()


def _simulate(traci, task, rng) -> dict:
    route_cache: dict[tuple, str] = {}
    driver_of: dict[str, str] = {}
    for trip in task["trips"]:
        key = (trip["from"], trip["to"])
        if key not in route_cache:
            edges = traci.simulation.findRoute(*key).edges
            if not edges:
                continue
            rid = f"r{len(route_cache)}"
            traci.route.add(rid, edges)
            route_cache[key] = rid
        traci.vehicle.add(trip["veh_id"], route_cache[key], typeID=trip["vtype"],
                          depart=f"{trip['depart']:.1f}", departLane="best", departSpeed="max")
        driver_of[trip["veh_id"]] = trip["driver_id"]

    incidents = [dict(i, affected=set(), length=traci.lane.getLength(f"{i['edge']}_0")) for i in task["incidents"]]

    # veh -> [detector, current_edge, traversals, trip stats (named drivers only)]
    state: dict[str, list] = {}
    restore: list[tuple[float, str]] = []  # (time, veh) to reset speed mode after an incident stop
    traversals, events, trips = [], [], []
    date = task["date"]
    count_lanes = task.get("count_lanes", False)

    def finish(vid):
        st = state.pop(vid, None)
        if st is None:
            return
        for edge, enter, count in st[2]:
            traversals.append((vid, driver_of.get(vid, ""), edge, date, int(enter // 3600),
                               int(enter % 3600 // 60), count))
        s = st[3]
        if s is not None and s["n"] > 10:
            n = s["n"]
            mean_v = s["sv"] / n
            trips.append({
                "trip_id": vid, "driver_id": driver_of[vid], "date": date, "window": task["window"],
                "depart_s": round(s["t0"], 1), "duration_s": round(s["t_last"] - s["t0"], 1),
                "distance_m": round(s["dist"], 1), "traversals": len(st[2]),
                "hard_brakes": sum(c for _, _, c in st[2]), "moderate_brakes": s["moderate"],
                "max_decel": round(s["max_decel"], 3),
                "accel_rms": round((s["saa"] / max(s["na"], 1)) ** 0.5, 4),
                "mean_speed": round(mean_v, 3),
                "speed_cv": round(max(s["svv"] / n - mean_v**2, 0) ** 0.5 / max(mean_v, 0.1), 4),
                **({"lane_changes": s["lc"]} if count_lanes else {}),
            })

    t = task["begin"]
    while t < task["end"]:
        traci.simulationStep()
        t = traci.simulation.getTime()

        for vid in traci.simulation.getDepartedIDList():
            named = bool(driver_of.get(vid))
            traci.vehicle.subscribe(vid, [VAR_SPEED, VAR_ROAD_ID] + ([VAR_LANE_INDEX] if count_lanes and named else []))
            stats = None
            if driver_of.get(vid):
                stats = {"t0": t, "t_last": t, "n": 0, "sv": 0.0, "svv": 0.0, "saa": 0.0, "na": 0,
                         "dist": 0.0, "last_v": None, "max_decel": 0.0, "moderate": 0,
                         "mod_det": HardBrakeDetector(start_decel=2.0, end_decel=1.0),
                         "lc": 0, "lane": None, "lane_road": None}
            state[vid] = [HardBrakeDetector(), None, [], stats]

        for vid, r in traci.vehicle.getAllSubscriptionResults().items():
            st = state.get(vid)
            if st is None:
                continue
            road = r[VAR_ROAD_ID]
            v = r[VAR_SPEED]
            if road and road[0] != ":" and road != st[1]:
                st[1] = road
                st[2].append([road, t, 0])
            if count_lanes and st[3] is not None and VAR_LANE_INDEX in r and road and road[0] != ":":
                s3 = st[3]   # a lane change = new lane index on the same (non-junction) edge
                if s3["lane_road"] == road and s3["lane"] is not None and r[VAR_LANE_INDEX] != s3["lane"]:
                    s3["lc"] += 1
                s3["lane"], s3["lane_road"] = r[VAR_LANE_INDEX], road
            ev = st[0].on_sample(t, v)
            s = st[3]
            if s is not None:
                s["n"] += 1
                s["sv"] += v
                s["svv"] += v * v
                if s["last_v"] is not None:
                    dt = t - s["t_last"]
                    if dt > 0:
                        s["dist"] += (v + s["last_v"]) / 2 * dt
                        if v > 0.5 or s["last_v"] > 0.5:   # skip standstill so queues don't dilute
                            a = (v - s["last_v"]) / dt
                            s["saa"] += a * a
                            s["na"] += 1
                s["last_v"] = v
                s["t_last"] = t
                if st[0].last_accel is not None and -st[0].last_accel > s["max_decel"]:
                    s["max_decel"] = -st[0].last_accel
                if s["mod_det"].on_sample(t, v) is not None:
                    s["moderate"] += 1
            if ev is not None and st[2]:
                # attribute to the traversal that contains the event start
                for trav in reversed(st[2]):
                    if trav[1] <= ev.start + 1e-9:
                        trav[2] += 1
                        events.append((vid, driver_of.get(vid, ""), trav[0], date, ev.start,
                                       ev.start_speed, ev.end_speed, ev.peak_decel, ev.duration))
                        break

        for inc in incidents:
            if inc["t0"] <= t < inc["t1"]:
                for vid in traci.edge.getLastStepVehicleIDs(inc["edge"]):
                    if vid in inc["affected"]:
                        continue
                    # React at the obstacle (middle of the edge), only if actually moving.
                    if traci.vehicle.getLanePosition(vid) < 0.4 * inc["length"]:
                        continue
                    inc["affected"].add(vid)
                    v = traci.vehicle.getSpeed(vid)
                    if v > 6.0 and rng.random() < inc["p"]:
                        duration = max((v - 1.0) / INCIDENT_DECEL, 0.4)
                        # Lift the comfortable-decel limit so the stop is really abrupt.
                        traci.vehicle.setSpeedMode(vid, SPEED_MODE_NO_DECEL_LIMIT)
                        traci.vehicle.slowDown(vid, 1.0, duration)
                        restore.append((t + duration + 1.0, vid))
                        inc["slowed"] = inc.get("slowed", 0) + 1

        while restore and restore[0][0] <= t:
            _, vid = restore.pop(0)
            if vid in state:
                traci.vehicle.setSpeedMode(vid, SPEED_MODE_DEFAULT)

        for vid in traci.simulation.getArrivedIDList():
            finish(vid)

    for vid in list(state):
        finish(vid)

    trav = pd.DataFrame(traversals, columns=["trip_id", "driver_id", "segment_id", "date", "hour", "minute",
                                             "hard_brakes"])
    ev = pd.DataFrame(events, columns=["trip_id", "driver_id", "segment_id", "date", "t", "start_speed",
                                       "end_speed", "peak_decel", "duration"])
    inc_stats = [{"edge": i["edge"], "seen": len(i["affected"]), "slowed": i.get("slowed", 0)} for i in incidents]
    return {"traversals": trav, "events": ev, "trips": pd.DataFrame(trips), "date": date, "begin": task["begin"],
            "incident_stats": inc_stats}
