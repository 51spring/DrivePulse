"""Ablation: which part of the context makes the baseline work?

The expected number of hard brakes in a (segment, time band, date) cell is computed in
increasingly specific ways, all leave-one-date-out (the cell's own date never feeds its
expectation):

  absolute   no baseline: rank cells by raw count (TMAP-style "most hard brakes")
  network    city-wide hard-brake rate             x passes   ("above the city average?")
  time       city-wide rate of that time band      x passes   ("busy hour?")
  segment    that segment's rate at all times      x passes   ("this road is always bad?")
  seg x time that segment in that time band, raw   x passes
  ours       seg x time shrunk toward segment x band factor (baseline.cell_baseline)

Segments: same Poisson test + ranking for every variant, scored on incidents that actually
changed driving. Drivers: each variant's per-pass expectation summed per driver -> SIR.

    python ablation.py            # all runs in out/exp, writes docs/results/ablation.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baseline import add_context, benjamini_hochberg, cell_baseline, poisson_upper_p  # noqa: E402
from evaluate import _key, effective_incidents, incident_cells  # noqa: E402
from experiments import AREA_KO, OUT, RUNS  # noqa: E402

VARIANTS = ["network", "time", "segment", "seg_time_raw", "ours"]
LABEL = {
    "absolute": "절대: 급정거 횟수 순 (기준선 없음)",
    "network": "도시 전체 평균 대비",
    "time": "시간대만 (모든 구간 공통)",
    "segment": "구간만 (모든 시간 공통)",
    "seg_time_raw": "구간 × 시간대 (축소 보정 없음)",
    "ours": "**구간 × 시간대 + 축소 보정 (제안)**",
}


def expected_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pass expected rate of every variant for each (segment, bucket, date) cell."""
    cells = cell_baseline(df)  # ours: base_rate, plus exposure/count
    cols = ["exposure", "count"]
    tot = cells[cols].sum()
    day = cells.groupby("date")[cols].transform("sum")
    b_all = cells.groupby("bucket")[cols].transform("sum")
    b_day = cells.groupby(["bucket", "date"])[cols].transform("sum")
    s_all = cells.groupby("segment_id")[cols].transform("sum")
    s_day = cells.groupby(["segment_id", "date"])[cols].transform("sum")
    sb_all = cells.groupby(["segment_id", "bucket"])[cols].transform("sum")

    def rate(all_, part):
        e = all_["exposure"] - part["exposure"]
        y = all_["count"] - part["count"]
        return (y / e.where(e > 0)).astype(float)

    cells["r_network"] = (tot["count"] - day["count"]) / (tot["exposure"] - day["exposure"])
    cells["r_time"] = rate(b_all, b_day)
    cells["r_segment"] = rate(s_all, s_day)
    cells["r_seg_time_raw"] = rate(sb_all, cells[cols]).fillna(cells["r_segment"])
    cells["r_ours"] = cells["base_rate"]
    for v in VARIANTS:
        cells[f"r_{v}"] = cells[f"r_{v}"].fillna(cells["r_network"]).clip(lower=1e-6)
    return cells


def flags_for(cells: pd.DataFrame, expected: pd.Series, min_exposure=10, alpha=0.05, min_ratio=2.0, min_count=3):
    p = pd.Series(poisson_upper_p(cells["count"].to_numpy(), expected.to_numpy()), index=cells.index)
    tested = cells["exposure"] >= min_exposure
    q = pd.Series(1.0, index=cells.index)
    q[tested] = benjamini_hochberg(p[tested].to_numpy())
    flagged = tested & (q < alpha) & (cells["count"] >= min_count) & (cells["count"] / expected >= min_ratio)
    return p, flagged


def evaluate_run(folder: Path, min_driver_traversals: int = 50) -> dict:
    trav = pd.read_csv(folder / "traversals.csv")
    truth_inc = pd.read_csv(folder / "truth_incidents.csv")
    truth_drv = pd.read_csv(folder / "truth_drivers.csv")
    df = add_context(trav)
    cells = expected_rates(df)

    inc_of = incident_cells(truth_inc, None)
    effective = effective_incidents(folder, truth_inc, df["date"].nunique())
    k = len(truth_inc)
    found_eff = lambda picked: len({inc_of[c] for c in _key(picked) if c in inc_of} & effective) / max(len(effective), 1)
    tested = cells["exposure"] >= 10

    out = {"k_eff": len(effective)}
    out["seg_absolute"] = found_eff(cells[tested].nlargest(k, "count"))
    ours_flagged = None
    for v in VARIANTS:
        expected = cells[f"r_{v}"] * cells["exposure"]
        p, flagged = flags_for(cells, expected)
        out[f"seg_{v}"] = found_eff(cells[tested].assign(p=p[tested]).nsmallest(k, "p"))
        fl = _key(cells[flagged])
        out[f"seg_{v}_flag_p"] = len(fl & set(inc_of)) / len(fl) if fl else 0.0
        out[f"seg_{v}_flag_n"] = len(fl)
        if v == "ours":
            ours_flagged = flagged

    # ---- drivers: same flagged road anomalies excluded for every variant
    keys = ["segment_id", "bucket", "date"]
    t = df[df["driver_id"] != "bg"].merge(
        cells[keys + [f"r_{v}" for v in VARIANTS]].assign(flagged=ours_flagged.to_numpy()), on=keys, how="left")
    t = t[~t["flagged"].fillna(False).astype(bool)]
    agg = {"traversals": ("hard_brakes", "size"), "observed": ("hard_brakes", "sum")}
    agg.update({f"e_{v}": (f"r_{v}", "sum") for v in VARIANTS})
    d = t.groupby("driver_id").agg(**agg).reset_index().merge(truth_drv[["driver_id", "label"]], on="driver_id")
    d = d[d["traversals"] >= min_driver_traversals]
    d["risky"] = d["label"] != "normal"
    d["route"] = d["e_ours"] / d["traversals"]
    kd = int(d["risky"].sum())
    normal = ~d["risky"]
    d["s_absolute"] = d["observed"] / d["traversals"]
    out["drv_absolute"] = d.nlargest(kd, "s_absolute")["risky"].mean()
    out["corr_absolute"] = d.loc[normal, ["s_absolute", "route"]].corr().iloc[0, 1]
    for v in VARIANTS:
        d[f"s_{v}"] = (d["observed"] + 5) / (d[f"e_{v}"] + 5)
        out[f"drv_{v}"] = d.nlargest(kd, f"s_{v}")["risky"].mean()
        out[f"corr_{v}"] = d.loc[normal, [f"s_{v}", "route"]].corr().iloc[0, 1]
    return out


def main():
    rows = []
    for name, (area, seed, _, strength) in RUNS.items():
        folder = OUT / name
        if not (folder / "traversals.csv").exists():
            continue
        r = evaluate_run(folder)
        r.update(run=name, area=area, seed=seed, strength=strength)
        rows.append(r)
        print(f"done {name}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "ablation.csv", index=False)

    main_runs = res[res["strength"] == 1.0]
    m, s = main_runs.mean(numeric_only=True), main_runs.std(numeric_only=True)
    variants = ["absolute"] + VARIANTS
    lines = [
        "# 중간 단계 비교 (Ablation): 기준선의 어느 부분이 효과를 내나", "",
        f"SUMO, 실제 한국 지도(강남 3회, 판교 2회), 28일. 값은 {len(main_runs)}회 평균 ± 표준편차. "
        "모든 기준선은 해당 날짜를 빼고 계산한다.", "",
        "## 구간 돌발: 실제로 급정거를 늘린 돌발 중 상위 K칸으로 찾은 비율", "",
        "| 기준선 | 적중률 | 자체 판정 정밀도 (판정 수) |", "|---|---|---|",
    ]
    for v in variants:
        hit = f"{m[f'seg_{v}']:.2f} ± {s[f'seg_{v}']:.2f}"
        flag = "-" if v == "absolute" else f"{m[f'seg_{v}_flag_p']:.2f} ({m[f'seg_{v}_flag_n']:.0f}건)"
        lines.append(f"| {LABEL[v]} | {hit} | {flag} |")
    lines += ["", "## 위험 운전자: 상위 K명 적중률과 길 효과", "",
              "| 기준선 | 적중률 | 정상 운전자 점수와 길 혼잡도의 상관 |", "|---|---|---|"]
    for v in variants:
        lines.append(f"| {LABEL[v]} | {m[f'drv_{v}']:.2f} ± {s[f'drv_{v}']:.2f} | "
                     f"{m[f'corr_{v}']:.2f} ± {s[f'corr_{v}']:.2f} |")
    sens = res[(res["area"] == "gangnam") & (res["seed"] == 3)].sort_values("strength")
    if len(sens) > 1:
        lines += ["", "## 민감도 (강남 시드 3): 위험 운전자 적중률", "",
                  "| 성향 강도 | " + " | ".join(LABEL[v].strip("*") for v in variants) + " |",
                  "|---" * (len(variants) + 1) + "|"]
        for r in sens.itertuples():
            lines.append(f"| {r.strength:g} | " + " | ".join(f"{getattr(r, f'drv_{v}'):.2f}" for v in variants) + " |")
    lines += ["", "## 실험별 값", "", "| 실험 | 효과 있던 돌발 | " + " | ".join(f"구간 {v}" for v in variants) + " |",
              "|---" * (len(variants) + 2) + "|"]
    for r in res.itertuples():
        lines.append(f"| {AREA_KO[r.area]} s{r.seed} 강도{r.strength:g} | {r.k_eff} | "
                     + " | ".join(f"{getattr(r, f'seg_{v}'):.2f}" for v in variants) + " |")
    text = "\n".join(lines) + "\n"
    out = HERE.parent / "docs" / "results" / "ablation.md"
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
