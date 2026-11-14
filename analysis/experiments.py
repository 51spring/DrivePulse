"""Stage-1 robustness experiments: repeat over seeds and Korean areas, plus a sensitivity
sweep over how different risky drivers are (style strength).

    python experiments.py run        # generate missing datasets (slow: ~10 min each)
    python experiments.py report     # evaluate all and write out/exp/summary.md + figure

Each run is a 28-day SUMO dataset in out/exp/<name>/.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "exp"

# name: (area, seed, population seed, style strength)
RUNS = {
    "gangnam_s3": ("gangnam", 3, 11, 1.0),
    "gangnam_s4": ("gangnam", 4, 12, 1.0),
    "gangnam_s5": ("gangnam", 5, 13, 1.0),
    "pangyo_s3": ("pangyo", 3, 11, 1.0),
    "pangyo_s4": ("pangyo", 4, 12, 1.0),
    "gangnam_s3_str0.5": ("gangnam", 3, 11, 0.5),
    "gangnam_s3_str0.25": ("gangnam", 3, 11, 0.25),
}
AREA_KO = {"gangnam": "강남", "pangyo": "판교"}


def run_all(days: int = 28, workers: int = 10):
    for name, (area, seed, pop_seed, strength) in RUNS.items():
        folder = OUT / name
        if (folder / "traversals.csv").exists():
            print(f"skip {name} (exists)")
            continue
        print(f"=== {name} ===", flush=True)
        subprocess.run([sys.executable, str(HERE / "sumo_sim" / "generate.py"), "--network", "osm",
                        "--osm-area", area, "--seed", str(seed), "--pop-seed", str(pop_seed),
                        "--style-strength", str(strength), "--days", str(days),
                        "--workers", str(workers), "--out", str(folder)], check=True)


def evaluate_all() -> pd.DataFrame:
    sys.path.insert(0, str(HERE))
    from evaluate import evaluate

    rows = []
    for name, (area, seed, pop_seed, strength) in RUNS.items():
        folder = OUT / name
        if not (folder / "traversals.csv").exists():
            continue
        cache = folder / "metrics.json"
        if cache.exists():
            rows.append(json.loads(cache.read_text(encoding="utf-8")))
            continue
        df, cells, drivers, _, seg, drv = evaluate(folder)
        normal_rate = drivers.loc[drivers["label"] == "normal", "naive_rate_per100"].mean()
        risky_rate = drivers.loc[drivers["label"] != "normal", "naive_rate_per100"].mean()
        row = {
            "run": name, "area": area, "seed": seed, "strength": strength,
            "traversals": len(df), "normal_per100": normal_rate, "risky_over_normal": risky_rate / normal_rate,
            "seg_count": seg["count"], "seg_context": seg["context"], "seg_scan": seg.get("scan"),
            "seg_k": seg["k"], "seg_k_eff": seg["k_eff"],
            "seg_context_eff": seg["context_eff"], "seg_scan_eff": seg.get("scan_eff"),
            "seg_flag_p": seg["flag_p"], "seg_flag_r": seg["flag_r"],
            "drv_naive": drv["naive"], "drv_context": drv["context"],
            "drv_naive_fp_congested": drv["naive_fp_congested"], "drv_ctx_fp_congested": drv["ctx_fp_congested"],
            "corr_naive_route": drv["corr_naive_route"], "corr_sir_route": drv["corr_sir_route"],
        }
        cache.write_text(json.dumps(row, ensure_ascii=False, default=float), encoding="utf-8")
        rows.append(row)
        print(f"evaluated {name}", flush=True)
    return pd.DataFrame(rows)


def report(res: pd.DataFrame):
    full = res[res["strength"] == 1.0]
    lines = ["# 1단계 반복·민감도 실험 결과", "",
             "모든 실험: SUMO, 실제 한국 지도(OpenStreetMap), 28일, 운전자 300명 + 배경 교통, 돌발 약 20건.", "",
             "## 반복 실험 (지역 × 시드)", "",
             "| 실험 | 정상 운전자 급정거/100회 | 위험/정상 배율 | 위험 운전자: 절대 → 맥락 | 절대 점수 오탐 중 혼잡경로 | 효과 있던 돌발: 절대 → 맥락 → 스캔 |",
             "|---|---|---|---|---|---|"]
    for r in full.itertuples():
        lines.append(
            f"| {AREA_KO[r.area]} 시드 {r.seed} | {r.normal_per100:.1f} | {r.risky_over_normal:.2f}배 | "
            f"{r.drv_naive:.2f} → **{r.drv_context:.2f}** | {r.drv_naive_fp_congested:.0%} | "
            f"{r.seg_count:.2f} → **{r.seg_context_eff:.2f}** → {r.seg_scan_eff:.2f} ({r.seg_k_eff}/{r.seg_k}건) |")
    if len(full) > 1:
        m, s = full.mean(numeric_only=True), full.std(numeric_only=True)
        lines += ["", f"- 평균 (± 표준편차, {len(full)}회): 위험 운전자 적중률 절대 {m.drv_naive:.2f} ± {s.drv_naive:.2f}, "
                      f"맥락 **{m.drv_context:.2f} ± {s.drv_context:.2f}**; "
                      f"효과 있던 돌발 적중률 맥락 **{m.seg_context_eff:.2f} ± {s.seg_context_eff:.2f}**, "
                      f"스캔 {m.seg_scan_eff:.2f} ± {s.seg_scan_eff:.2f}"]

    sens = res[(res["area"] == "gangnam") & (res["seed"] == 3)].sort_values("strength")
    if len(sens) > 1:
        lines += ["", "## 민감도: 위험 운전자가 정상과 얼마나 다를 때 찾을 수 있나 (강남 시드 3)", "",
                  "| 성향 강도 | 위험/정상 급정거 배율 | 절대 점수 | 맥락 보정 |", "|---|---|---|---|"]
        for r in sens.itertuples():
            lines.append(f"| {r.strength:.2f} | {r.risky_over_normal:.2f}배 | {r.drv_naive:.2f} | **{r.drv_context:.2f}** |")
        plot_sensitivity(sens, OUT / "sensitivity.png")
        lines += ["", "![민감도](sensitivity.png)"]
    text = "\n".join(lines) + "\n"
    (OUT / "summary.md").write_text(text, encoding="utf-8")
    res.to_csv(OUT / "metrics.csv", index=False)
    print(text)


def plot_sensitivity(sens: pd.DataFrame, path: Path):
    for font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if any(font in f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = sens["risky_over_normal"]
    ax.plot(x, sens["drv_naive"], "o-", color="#9aa5b1", lw=2, label="절대 점수 (100회당 급정거)")
    ax.plot(x, sens["drv_context"], "o-", color="#1f6feb", lw=2, label="맥락 보정 SIR")
    ax.axhline(83 / 300, color="#555", ls=":", lw=1, label="무작위로 고를 때")
    for r in sens.itertuples():
        ax.annotate(f"강도 {r.strength:g}", (r.risky_over_normal, r.drv_context), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("위험 운전자 급정거율 / 정상 운전자 급정거율")
    ax.set_ylabel("위험 운전자 precision@K")
    ax.set_ylim(0, 1.05)
    ax.set_title("성향 차이가 작아질 때 두 방법의 적중률 (강남)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "run":
        run_all()
    report(evaluate_all())
