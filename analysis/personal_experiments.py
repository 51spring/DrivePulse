"""Evaluates the personal commute baseline on every dataset in out/personal/ and writes
docs/results/personal.md + docs/images/personal_*.png.

    python personal_experiments.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from personal import METHOD_KO, METHODS, plot_example, run  # noqa: E402

OUT = HERE / "out" / "personal"
DOCS = HERE.parent / "docs"
AREA_KO = {"gangnam": "강남", "pangyo": "판교"}


def describe(name: str) -> str:
    parts = name.split("_")
    area = AREA_KO.get(parts[1], parts[1])
    seed = parts[2].lstrip("s")
    strength = next((p[1:] for p in parts[3:] if p.startswith("a")), "1.0")
    return f"{area} 시드 {seed}" + ("" if strength == "1.0" else f" (변화 강도 {strength})")


def main():
    rows, examples = [], {}
    for folder in sorted(OUT.glob("p_*")):
        if not (folder / "trips.csv").exists():
            continue
        cache = folder / "personal_metrics.json"
        if cache.exists():
            row = json.loads(cache.read_text(encoding="utf-8"))
        else:
            day, drift, _, daily = run(folder)
            row = {"run": folder.name, **day, **drift}
            cache.write_text(json.dumps(row, ensure_ascii=False, default=float), encoding="utf-8")
            examples[folder.name] = daily
        rows.append(row)
        print(f"done {folder.name}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "personal_metrics.csv", index=False)

    full = res[~res["run"].str.contains("_a0")]
    m, s = full.mean(numeric_only=True), full.std(numeric_only=True)
    L = ["# 개인 출퇴근 기준선: 하루 이상과 장기 변화 탐지", "",
         f"SUMO, 실제 한국 지도, 28일, 운전자 300명(정상 70%·졸음 15%·공격적 15%) + 배경 교통. "
         f"각 실험에 하루 이상 60건(피로/조급), 장기 변화 운전자 20명을 심었다. "
         f"기본 강도 실험 {len(full)}회 평균 ± 표준편차.", "",
         "## 1. 하루 이상: 오늘 출퇴근이 '평소의 나'와 다른가", "",
         "평가 대상은 같은 경로·시간대 기록이 3회 이상 쌓인 뒤의 출퇴근이다. "
         "AUC는 이상 운행을 정상 운행보다 높게 점수 매길 확률이고(0.5 = 무작위), "
         "적중률은 점수 상위 K개(K = 심은 이상 수) 중 진짜 이상의 비율이다.", "",
         "| 방법 | AUC | 상위 K 적중률 | 정상 운전자 AUC | 졸음·공격적 운전자 AUC | 피로 AUC | 조급 AUC |",
         "|---|---|---|---|---|---|---|"]
    for meth in METHODS:
        risky = np.nanmean([m.get(f"auc_{meth}_drowsy", np.nan), m.get(f"auc_{meth}_aggressive", np.nan)])
        L.append(f"| {METHOD_KO[meth]} | {m[f'auc_{meth}']:.2f} ± {s[f'auc_{meth}']:.2f} | "
                 f"{m[f'p@k_{meth}']:.2f} | {m[f'auc_{meth}_normal']:.2f} | {risky:.2f} | "
                 f"{m[f'auc_{meth}_fatigue']:.2f} | {m[f'auc_{meth}_hurried']:.2f} |")
    L += ["", "## 2. 장기 변화: 운전이 몇 주에 걸쳐 바뀌고 있나, 언제부터", "",
          "모든 방법의 경보 기준을 '변화 없는 운전자 중 10%에서 경보'가 되도록 똑같이 맞춘 뒤 비교했다. "
          "기준 기간은 각 운전자의 첫 5 근무일이다.", "",
          "| 방법 | 탐지율 | 탐지까지 걸린 근무일 (중앙값) | 변화 시작 전 경보 |", "|---|---|---|---|"]
    for meth in METHODS:
        L.append(f"| {METHOD_KO[meth]} | {m[f'detect_{meth}']:.2f} ± {s[f'detect_{meth}']:.2f} | "
                 f"{m[f'delay_{meth}']:.1f} | {m[f'early_{meth}']:.1f}명 |")
    weak = res[res["run"].str.contains("_a0")]
    if len(weak):
        L += ["", "## 3. 하루 이상이 약할 때 (강도 0.5, 강남 시드 3)", "",
              "피로·조급의 효과를 절반으로 줄인 하루 이상만 다시 심었다(장기 변화는 그대로라 표에서 뺐다).", "",
              "| 방법 | 하루 이상 AUC | 상위 K 적중률 | 졸음·공격적 운전자 AUC |", "|---|---|---|---|"]
        w = weak.iloc[0]
        for meth in METHODS:
            risky = np.nanmean([w.get(f"auc_{meth}_drowsy", np.nan), w.get(f"auc_{meth}_aggressive", np.nan)])
            L.append(f"| {METHOD_KO[meth]} | {w[f'auc_{meth}']:.2f} | {w[f'p@k_{meth}']:.2f} | {risky:.2f} |")
    L += ["", "## 실험별 값", "", "| 실험 | 이상 수 | " + " | ".join(f"AUC {METHOD_KO[x]}" for x in METHODS) + " |",
          "|---" * (len(METHODS) + 2) + "|"]
    for r in res.itertuples():
        L.append(f"| {describe(r.run)} | {r.n_anomalies} | " +
                 " | ".join(f"{getattr(r, f'auc_{x}'):.2f}" for x in METHODS) + " |")
    L += ["", "![하루 이상과 장기 변화](../images/personal_summary.png)", "",
          "![장기 변화 예시](../images/personal_drift_example.png)", ""]
    (DOCS / "results" / "personal.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))

    plot_summary(full, DOCS / "images" / "personal_summary.png")
    first = sorted(OUT.glob("p_gangnam_s3"))
    if first:
        from personal import daily_series, drift_statistics, load
        f = first[0]
        daily = examples.get(f.name)
        if daily is None:
            t = load(f).merge(pd.read_csv(f / "truth_drivers.csv")[["driver_id", "label"]], on="driver_id")
            daily = drift_statistics(daily_series(t))
        plot_example(daily, pd.read_csv(f / "truth_drift.csv"), DOCS / "images" / "personal_drift_example.png")


def plot_summary(full: pd.DataFrame, path: Path):
    for font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if any(font in f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False
    m, s = full.mean(numeric_only=True), full.std(numeric_only=True)
    colors = ["#c7ccd4", "#9aa5b1", "#7aa7e6", "#1f6feb", "#d64545"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(METHODS))
    labels = [METHOD_KO[k].replace(" + ", "\n+ ").replace(" (", "\n(") for k in METHODS]

    ax = axes[0]
    ax.bar(x, [m[f"auc_{k}"] for k in METHODS], yerr=[s[f"auc_{k}"] for k in METHODS], color=colors, capsize=4)
    ax.axhline(0.5, color="#555", ls=":", lw=1)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.set_title("하루 이상 탐지 (AUC)")

    ax = axes[1]
    w = 0.38
    ax.bar(x - w / 2, [m[f"auc_{k}_normal"] for k in METHODS], w, color="#9aa5b1", label="정상 운전자")
    ax.bar(x + w / 2, [np.nanmean([m[f"auc_{k}_drowsy"], m[f"auc_{k}_aggressive"]]) for k in METHODS], w,
           color="#d64545", label="원래 거친 운전자 (졸음·공격적)")
    ax.axhline(0.5, color="#555", ls=":", lw=1)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0.3, 1.0)
    ax.set_title("운전자의 기본 성향별 하루 이상 탐지 (AUC)")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(x, [m[f"detect_{k}"] for k in METHODS], yerr=[s[f"detect_{k}"] for k in METHODS], color=colors, capsize=4)
    for i, k in enumerate(METHODS):
        d = m[f"delay_{k}"]
        if not np.isnan(d):
            ax.text(i, m[f"detect_{k}"] + 0.04, f"{d:.0f}일", ha="center", fontsize=9)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title("장기 변화 탐지율 (오경보 10% 기준, 숫자 = 탐지까지 근무일)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
