"""SUMO scenario: road network, persistent driver population and per-driver vehicle types.

Network: signalised grid (netgenerate), 2 lanes, 60 km/h, with entry/exit roads on the
border. Each SUMO edge is one road segment for the analysis.

Drivers keep the same driving style (vType) and commute every day, so driver-level
anomalies can be measured across days. Styles (carFollowModel EIDM):
  normal     : treaction ~0.5 s, tau ~1.1 s
  drowsy     : late reaction (treaction 1.2-1.6 s), large perception errors
  aggressive : short headway (tau ~0.6 s), speeding, strong accel/decel, pushy lane changes
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np
import pandas as pd
import sumo
import sumolib

SUMO_BIN = Path(sumo.SUMO_HOME) / "bin"

LABEL_SHARE = {"normal": 0.70, "drowsy": 0.15, "aggressive": 0.15}


GRID_ARGS = [
    "--grid", "--grid.x-number", "6", "--grid.y-number", "4",
    "--grid.x-length", "300", "--grid.y-length", "250", "--grid.attach-length", "200",
    "--default.lanenumber", "2", "--default.speed", "16.67",
    "--default.junctions.type", "traffic_light", "--no-turnarounds", "true", "--seed", "1",
]

# Road classes for the "mixed" network (row index of the grid -> class)
ROAD_CLASS = {0: "expressway", 1: "arterial", 2: "arterial", 3: "local"}
CLASS_SPEC = {  # speed m/s, lanes, priority
    "expressway": (25.0, 3, 10),
    "arterial": (16.67, 2, 5),
    "local": (11.11, 1, 2),
    "connector": (16.67, 2, 3),   # vertical streets
}


def build_network(out_dir: Path, kind: str = "grid", osm_area: str = "gangnam") -> Path:
    """kind="grid": every road a signalised 60 km/h arterial (homogeneous risk).
    kind="mixed": row 0 expressway (90 km/h, 3 lanes, no signals, side roads yield),
    rows 1-2 signalised arterials, row 3 local road (40 km/h, 1 lane) -> route risk varies.
    kind="osm": a real Korean road network from OpenStreetMap (see osm_korea.py).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if kind == "osm":
        from osm_korea import PRESETS, build_osm_network
        *bbox, _ = PRESETS[osm_area]
        return build_osm_network(tuple(bbox), out_dir, osm_area)
    net = out_dir / f"{kind}.net.xml"
    if net.exists():
        return net
    if kind == "grid":
        subprocess.run([str(SUMO_BIN / "netgenerate"), *GRID_ARGS, "-o", str(net)], check=True, capture_output=True)
        return net
    if kind != "mixed":
        raise ValueError(kind)

    import xml.etree.ElementTree as ET
    plain = out_dir / "plain"
    plain.mkdir(exist_ok=True)
    subprocess.run([str(SUMO_BIN / "netgenerate"), *GRID_ARGS, "--plain-output-prefix", str(plain / "g"),
                    "-o", str(plain / "g.net.xml")], check=True, capture_output=True)

    def row(node_id: str):
        """Grid row of a node: A0..F3 -> digit, left2/right2 -> digit, bottom*/top* -> None."""
        if node_id.startswith(("left", "right")) or (len(node_id) == 2 and node_id[0].isalpha()):
            return int(node_id[-1])
        return None

    nodes = ET.parse(plain / "g.nod.xml")
    for n in nodes.getroot().iter("node"):
        r = row(n.get("id"))
        if n.get("id").startswith(("left", "right", "bottom", "top")):
            n.set("type", "priority")
        elif r is not None and ROAD_CLASS[r] == "expressway":
            n.set("type", "priority")      # free-flowing, cross streets yield
        else:
            n.set("type", "traffic_light")
        n.attrib.pop("tl", None)

    edges = ET.parse(plain / "g.edg.xml")
    for e in edges.getroot().iter("edge"):
        r_from, r_to = row(e.get("from")), row(e.get("to"))
        cls = ROAD_CLASS[r_from] if (r_from is not None and r_from == r_to) else "connector"
        speed, lanes, prio = CLASS_SPEC[cls]
        e.set("speed", f"{speed:.2f}")
        e.set("numLanes", str(lanes))
        e.set("priority", str(prio))
        e.set("type", cls)
    nodes.write(plain / "m.nod.xml")
    edges.write(plain / "m.edg.xml")
    subprocess.run([str(SUMO_BIN / "netconvert"), "-n", str(plain / "m.nod.xml"), "-e", str(plain / "m.edg.xml"),
                    "--no-turnarounds", "true", "-o", str(net)], check=True, capture_output=True)
    return net


def road_classes(net_file: Path) -> dict[str, str]:
    """edge id -> road class (edge 'type'), '' for the plain grid."""
    net = sumolib.net.readNet(str(net_file))
    return {e.getID(): (e.getType().replace("highway.", "") or "arterial") for e in net.getEdges()}


def fringe_edges(net_file: Path):
    """Returns (sources, sinks) as {edge_id: border_node_id} dicts.

    Border node = dead end (connected to exactly one other node) lying near the edge of
    the network's bounding box. For OSM maps these are the roads cut at the map border,
    so interior dead ends are not used as trip origins.
    """
    net = sumolib.net.readNet(str(net_file))
    xmin, ymin, xmax, ymax = net.getBoundary()
    margin = 0.12 * min(xmax - xmin, ymax - ymin)
    sources, sinks = {}, {}
    for node in net.getNodes():
        neighbours = {e.getToNode().getID() for e in node.getOutgoing()} | \
                     {e.getFromNode().getID() for e in node.getIncoming()}
        if len(neighbours) != 1:
            continue
        x, y = node.getCoord()
        if min(x - xmin, xmax - x, y - ymin, ymax - y) > margin:
            continue
        for e in node.getOutgoing():
            sources[e.getID()] = node.getID()
        for e in node.getIncoming():
            sinks[e.getID()] = node.getID()
    return dict(sorted(sources.items())), dict(sorted(sinks.items()))


def _style_params(label: str, rng: np.random.Generator) -> dict:
    j = lambda m, s, lo, hi: float(np.clip(rng.normal(m, s), lo, hi))
    if label == "normal":
        return dict(accel=j(2.6, 0.2, 2.0, 3.2), decel=j(2.2, 0.2, 1.8, 2.6), tau=j(1.2, 0.15, 0.9, 1.6),
                    speedFactor=j(1.0, 0.05, 0.9, 1.1), minGap=2.5, treaction=j(0.5, 0.1, 0.3, 0.8),
                    sigmaerror=0.1, sigmagap=0.1, sigmaleader=0.05, lcAssertive=1.0)
    if label == "drowsy":
        return dict(accel=j(2.4, 0.2, 1.8, 3.0), decel=j(2.2, 0.2, 1.8, 2.6), tau=j(1.1, 0.15, 0.8, 1.5),
                    speedFactor=j(0.97, 0.05, 0.85, 1.1), minGap=2.5, treaction=j(1.2, 0.15, 0.9, 1.5),
                    sigmaerror=0.25, sigmagap=0.25, sigmaleader=0.15, lcAssertive=1.0)
    if label == "aggressive":
        return dict(accel=j(3.5, 0.3, 2.8, 4.2), decel=j(4.0, 0.3, 3.5, 4.5), tau=j(0.6, 0.1, 0.4, 0.8),
                    speedFactor=j(1.2, 0.05, 1.1, 1.3), minGap=1.2, treaction=j(0.4, 0.1, 0.2, 0.6),
                    sigmaerror=0.1, sigmagap=0.1, sigmaleader=0.05, lcAssertive=2.0)
    raise ValueError(label)


def blended_style(label: str, rng: np.random.Generator, strength: float = 1.0) -> dict:
    """Style parameters with the drowsy/aggressive deviation scaled by `strength`.

    strength=1: full style (as before); 0: indistinguishable from a normal driver.
    Used for sensitivity analysis: how different must risky drivers be before each
    method can find them?
    """
    target = _style_params(label, rng)
    if label == "normal" or strength >= 1.0:
        return target
    base = _style_params("normal", rng)
    return {k: base[k] + strength * (target[k] - base[k]) for k in target}


STATE_KINDS = ("fatigue", "hurried")


def style_variant(params: dict, kind: str, strength: float) -> dict:
    """A driver's own style changed by a temporary or gradual state.

    fatigue : slower reaction, larger perception errors, slightly slower (tired / drowsy day)
    hurried : shorter headway, faster, harder accel/brake, pushier lane changes (rushed day)
    strength 0 = unchanged, 1 = full effect. Used for both one-day anomalies and
    long-term drift (strength ramps up over weeks).
    """
    p = dict(params)
    s = float(np.clip(strength, 0.0, 1.5))
    if kind == "fatigue":
        p["treaction"] = p["treaction"] + 0.8 * s
        p["sigmaerror"] = p["sigmaerror"] + 0.25 * s
        p["sigmagap"] = p["sigmagap"] + 0.25 * s
        p["sigmaleader"] = p["sigmaleader"] + 0.15 * s
        p["speedFactor"] = p["speedFactor"] - 0.03 * s
    elif kind == "hurried":
        p["tau"] = max(p["tau"] * (1 - 0.4 * s), 0.35)
        p["speedFactor"] = p["speedFactor"] + 0.15 * s
        p["accel"] = p["accel"] + 0.8 * s
        p["decel"] = p["decel"] + 1.5 * s
        p["minGap"] = max(p["minGap"] - 1.0 * s, 1.0)
        p["lcAssertive"] = p["lcAssertive"] + 1.0 * s
    else:
        raise ValueError(kind)
    return p


def make_population(net_file: Path, n_drivers: int = 300, seed: int = 11,
                    style_strength: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sources, sinks = fringe_edges(net_file)
    src_ids, sink_ids = list(sources), list(sinks)
    net = sumolib.net.readNet(str(net_file))
    xy = lambda node_id: np.array(net.getNode(node_id).getCoord())

    def nearest(node_id: str, pool: dict) -> str:
        """Edge in pool whose border node is closest to node_id (exact match first).
        Real maps have one-way border roads, so the return trip may use a nearby node."""
        here = xy(node_id)
        return min(pool, key=lambda e: np.linalg.norm(xy(pool[e]) - here))

    labels = rng.choice(list(LABEL_SHARE), size=n_drivers, p=list(LABEL_SHARE.values()))
    rows = []
    for i, label in enumerate(labels):
        home = src_ids[rng.integers(len(src_ids))]
        work = sink_ids[rng.integers(len(sink_ids))]
        while sinks[work] == sources[home]:  # don't commute back to the same border point
            work = sink_ids[rng.integers(len(sink_ids))]
        rows.append({"driver_id": f"drv_{i:04d}", "label": label, "home": home, "work": work,
                     "pm_from": nearest(sinks[work], sources), "pm_to": nearest(sources[home], sinks),
                     **blended_style(label, rng, style_strength)})
    return pd.DataFrame(rows)


def _vtype_xml(type_id: str, p: dict) -> str:
    attrs = {
        "id": type_id, "carFollowModel": "EIDM", "length": "4.6",
        "emergencyDecel": "9.0", "sigma": "0",
        **{k: f"{p[k]:.3f}" for k in ("accel", "decel", "tau", "speedFactor", "minGap",
                                       "treaction", "sigmaerror", "sigmagap", "sigmaleader")},
        "lcAssertive": f"{p['lcAssertive']:.2f}",
    }
    return "    <vType " + " ".join(f"{k}={quoteattr(v)}" for k, v in attrs.items()) + "/>"


def write_vtypes(drivers: pd.DataFrame, path: Path, seed: int = 5, variants: dict | None = None) -> Path:
    """One vType per driver, extra per-trip variants {type_id: params}, and 'bg' types
    for anonymous background traffic."""
    rng = np.random.default_rng(seed)
    lines = ["<additional>"]
    for _, d in drivers.iterrows():
        lines.append(_vtype_xml(d["driver_id"], d.to_dict()))
    for type_id, p in (variants or {}).items():
        lines.append(_vtype_xml(type_id, p))
    for k in range(20):  # background: mostly normal, some mild variety
        label = rng.choice(["normal", "drowsy", "aggressive"], p=[0.8, 0.1, 0.1])
        lines.append(_vtype_xml(f"bg_{k:02d}", _style_params(label, rng)))
    lines.append("</additional>")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
