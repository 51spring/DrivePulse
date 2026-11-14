"""Real Korean road networks from OpenStreetMap, converted for SUMO.

Only areas inside South Korea are accepted. Roads kept: motorway, trunk, primary,
secondary, tertiary (+ their ramps/links) — the roads commuters actually drive; small
residential/service roads are dropped to keep the simulation fast.

    python sumo_sim/osm_korea.py --area gangnam            # preset
    python sumo_sim/osm_korea.py --bbox 37.55,126.97,37.57,127.00 --name jongno

Data: (c) OpenStreetMap contributors, ODbL.
"""
from __future__ import annotations

import argparse
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

import sumo

SUMO_HOME = Path(sumo.SUMO_HOME)
OVERPASS_URLS = [  # public Overpass instances, tried in order
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# South Korea incl. Jeju and Ulleung/Dokdo (lat_min, lon_min, lat_max, lon_max)
KOREA_BOUNDS = (33.0, 124.5, 38.7, 131.95)

PRESETS = {
    # name: (lat_min, lon_min, lat_max, lon_max, description)
    "gangnam": (37.494, 127.024, 37.512, 127.066, "서울 강남 테헤란로 (강남역~삼성역)"),
    "pangyo": (37.388, 127.090, 37.412, 127.125, "성남 판교 테크노밸리 (경부고속도로·분당내곡간 고속화도로 포함)"),
}

ROAD_FILTER = "^(motorway|trunk|primary|secondary|tertiary)(_link)?$"


def check_korea(bbox: tuple[float, float, float, float]) -> None:
    s, w, n, e = bbox
    ks, kw, kn, ke = KOREA_BOUNDS
    if not (s < n and w < e):
        raise ValueError(f"invalid bbox {bbox}: need lat_min<lat_max, lon_min<lon_max")
    if not (ks <= s and n <= kn and kw <= w and e <= ke):
        raise ValueError(f"bbox {bbox} is outside South Korea {KOREA_BOUNDS}; only Korean maps are supported")
    if (n - s) * (e - w) > 0.01:  # ~ 11 km x 9 km
        raise ValueError("bbox too large for a quick simulation; keep it under ~0.01 deg^2")


SHARED_CACHE = Path(__file__).resolve().parent.parent / "out" / "osm"


def download(bbox, out_file: Path) -> Path:
    check_korea(bbox)
    if out_file.exists():
        return out_file
    cached = SHARED_CACHE / out_file.name   # reuse instead of hitting Overpass again
    if cached.exists() and cached != out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_bytes(cached.read_bytes())
        return out_file
    s, w, n, e = bbox
    query = (f'[out:xml][timeout:120];(way["highway"~"{ROAD_FILTER}"]({s},{w},{n},{e}););'
             "(._;>;);out body;")
    body = urllib.parse.urlencode({"data": query}).encode()
    data, errors = b"", []
    for attempt in range(3):
        for url in OVERPASS_URLS:
            try:
                req = urllib.request.Request(url, data=body, headers={"User-Agent": "DrivePulse-research/0.1"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = r.read()
                break
            except Exception as e:  # 429/502/504 are common on public instances
                errors.append(f"{url}: {e}")
        if data:
            break
        time.sleep(10 * (attempt + 1))
    if not data:
        raise RuntimeError("Overpass download failed:\n" + "\n".join(errors))
    if b"<way" not in data:
        raise RuntimeError("Overpass returned no roads for this bbox")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(data)
    return out_file


# Korean defaults for OSM ways without `lanes` / `maxspeed` tags (most ways have neither).
# - Lanes: SUMO's typemap assumes 1 lane for secondary/tertiary roads, but Korean urban
#   secondary roads (e.g. 도곡로, 선릉로, 삼성로) have 2-3 lanes per direction; 1 lane
#   created artificial stop-and-go waves and ~1 hard brake per pass.
# - Speed: SUMO's default for primary roads is 100 km/h. Korea's "안전속도 5030" (2021)
#   sets 50 km/h on urban roads and 30 km/h on side streets; urban expressways ~80 km/h,
#   motorways 100 km/h. Explicit `maxspeed` tags in OSM still take precedence.
KOREA_LANE_DEFAULTS = """<types>
    <type id="highway.motorway"       priority="14" numLanes="3" speed="27.78" oneway="true"/>
    <type id="highway.motorway_link"  priority="9"  numLanes="1" speed="16.67" oneway="true"/>
    <type id="highway.trunk"          priority="13" numLanes="2" speed="22.22"/>
    <type id="highway.trunk_link"     priority="8"  numLanes="1" speed="13.89"/>
    <type id="highway.primary"        priority="12" numLanes="3" speed="13.89"/>
    <type id="highway.primary_link"   priority="7"  numLanes="1" speed="11.11"/>
    <type id="highway.secondary"      priority="11" numLanes="2" speed="13.89"/>
    <type id="highway.secondary_link" priority="6"  numLanes="1" speed="11.11"/>
    <type id="highway.tertiary"       priority="10" numLanes="2" speed="13.89"/>
    <type id="highway.tertiary_link"  priority="5"  numLanes="1" speed="8.33"/>
</types>
"""


def build_osm_network(bbox, out_dir: Path, name: str) -> Path:
    check_korea(bbox)
    out_dir.mkdir(parents=True, exist_ok=True)
    net = out_dir / f"{name}.net.xml"
    if net.exists():
        return net
    osm = download(bbox, out_dir / f"{name}.osm.xml")
    lane_types = out_dir / "korea_lanes.typ.xml"
    lane_types.write_text(KOREA_LANE_DEFAULTS, encoding="utf-8")
    subprocess.run([
        str(SUMO_HOME / "bin" / "netconvert"),
        "--osm-files", str(osm),
        "--type-files", f"{SUMO_HOME / 'data' / 'typemap' / 'osmNetconvert.typ.xml'},{lane_types}",
        "--keep-edges.by-vclass", "passenger",
        "--remove-edges.isolated", "true",
        "--keep-edges.components", "1",        # largest connected part only
        "--geometry.remove", "true",
        "--ramps.guess", "true",
        "--junctions.join", "true",
        "--tls.guess-signals", "true", "--tls.discard-simple", "true", "--tls.join", "true",
        "--roundabouts.guess", "true",
        "--output.street-names", "true",
        "--no-warnings", "true",
        "-o", str(net),
    ], check=True, capture_output=True, text=True)
    return net


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", choices=list(PRESETS), default="gangnam")
    ap.add_argument("--bbox", help="lat_min,lon_min,lat_max,lon_max (must be inside South Korea)")
    ap.add_argument("--name", help="network name when using --bbox")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "out" / "osm")
    args = ap.parse_args()
    if args.bbox:
        bbox = tuple(float(x) for x in args.bbox.split(","))
        name = args.name or "custom"
    else:
        *bbox, desc = PRESETS[args.area]
        name = args.area
        print(desc)
    net = build_osm_network(tuple(bbox), args.out, name)
    print(f"network: {net}")


if __name__ == "__main__":
    main()
