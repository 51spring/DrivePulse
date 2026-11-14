"""End-to-end demo on synthetic data with known ground truth.

    python run_demo.py [--out out/demo] [--seed 7]

Compares the context baseline with absolute (context-free) rules:
  * segments: rank cells by raw hard-brake count (naive) vs. Poisson p-value against baseline
  * drivers : rank by raw hard brakes per 100 traversals (naive, TMAP-style score)
              vs. shrunk SIR (observed / expected in the same segment x time context)
Metric: precision@K with K = number of planted anomalies, plus precision/recall of the
detector's own flags.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline import add_context, detect_driver_anomalies, detect_segment_anomalies
from simulate import simulate


def precision_recall(flagged: set, truth: set) -> tuple[float, float]:
    tp = len(flagged & truth)
    return (tp / len(flagged) if flagged else 0.0, tp / len(truth) if truth else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "out" / "demo")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    trav, truth_inc, truth_drv = simulate(seed=args.seed)
    trav.to_csv(args.out / "traversals.csv", index=False)
    truth_inc.to_csv(args.out / "truth_incidents.csv", index=False)
    truth_drv.to_csv(args.out / "truth_drivers.csv", index=False)

    df = add_context(trav)
    cells = detect_segment_anomalies(df)
    drivers = detect_driver_anomalies(df, cells).merge(truth_drv, on="driver_id")
    cells.to_csv(args.out / "segment_cells.csv", index=False)
    drivers.to_csv(args.out / "driver_scores.csv", index=False)

    # ---- segments
    key = lambda d: set(zip(d["segment_id"], d["bucket"], d["date"]))
    t_inc = key(truth_inc)
    k_inc = len(t_inc)
    tested = cells[cells["exposure"] >= 10]
    seg_ctx_p, seg_ctx_r = precision_recall(key(cells[cells["flagged"]]), t_inc)
    naive_abs = tested[tested["count"] >= 3]
    seg_abs_p, seg_abs_r = precision_recall(key(naive_abs), t_inc)
    seg_ctx_at_k = len(key(tested.nsmallest(k_inc, "p_value")) & t_inc) / k_inc
    seg_naive_at_k = len(key(tested.nlargest(k_inc, "count")) & t_inc) / k_inc

    # ---- drivers
    t_risky = set(drivers.loc[drivers["risky"], "driver_id"])
    k_drv = len(t_risky)
    drv_ctx_p, drv_ctx_r = precision_recall(set(drivers.loc[drivers["flagged"], "driver_id"]), t_risky)
    top_naive = drivers.nlargest(k_drv, "naive_rate_per100")
    top_ctx = drivers.nlargest(k_drv, "sir_shrunk")
    drv_naive_at_k = top_naive["risky"].mean()
    drv_ctx_at_k = top_ctx["risky"].mean()
    naive_fp = top_naive[~top_naive["risky"]]
    ctx_fp = top_ctx[~top_ctx["risky"]]

    corridor_share = truth_drv["corridor_commuter"].mean()
    lines = [
        "# 급정거 이상탐지 데모 (합성 데이터)",
        "",
        f"- 운전자 {len(truth_drv)}명, 구간 {df['segment_id'].nunique()}개, {df['date'].nunique()}일, 통과 {len(df):,}건, 급정거 {int(df['hard_brakes'].sum()):,}회",
        f"- 심어둔 정답: 구간 돌발 {k_inc}건 (급정거율 ×5), 위험 운전자 {k_drv}명 (×2.5~3.5)",
        f"- 혼잡 구간(급정거율 ×3) 출퇴근 운전자 비율: {corridor_share:.0%} (위험도와 무관)",
        "",
        "## 구간 이상 (구간 × 날짜 × 시간대)",
        "",
        "| 방법 | precision | recall | precision@K |",
        "|---|---|---|---|",
        f"| 절대 기준 (급정거 3회 이상) | {seg_abs_p:.2f} ({len(naive_abs)}건 플래그) | {seg_abs_r:.2f} | {seg_naive_at_k:.2f} (횟수 순) |",
        f"| 맥락 기준선 (Poisson + FDR) | {seg_ctx_p:.2f} ({int(cells['flagged'].sum())}건 플래그) | {seg_ctx_r:.2f} | {seg_ctx_at_k:.2f} (p값 순) |",
        "",
        f"## 위험 운전자 (K = {k_drv})",
        "",
        "| 방법 | precision@K | 오탐 중 혼잡구간 출퇴근자 | 자체 플래그 precision / recall |",
        "|---|---|---|---|",
        f"| 절대 점수 (100회 통과당 급정거) | {drv_naive_at_k:.2f} | {naive_fp['corridor_commuter'].mean() if len(naive_fp) else 0:.0%} ({len(naive_fp)}명 중) | - |",
        f"| 맥락 보정 SIR | {drv_ctx_at_k:.2f} | {ctx_fp['corridor_commuter'].mean() if len(ctx_fp) else 0:.0%} ({len(ctx_fp)}명 중) | {drv_ctx_p:.2f} / {drv_ctx_r:.2f} |",
        "",
    ]
    summary = "\n".join(lines)
    (args.out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)

    plot(drivers, cells, t_inc, args.out / "demo_results.png",
         {"seg_naive": seg_naive_at_k, "seg_ctx": seg_ctx_at_k, "drv_naive": drv_naive_at_k, "drv_ctx": drv_ctx_at_k})
    print(f"\n결과: {args.out}")


def plot(drivers, cells, t_inc, path, scores):
    for font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if any(font in f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    groups = [
        (~drivers["risky"] & ~drivers["corridor_commuter"], "#9aa5b1", "일반"),
        (~drivers["risky"] & drivers["corridor_commuter"], "#e8a33d", "일반 · 혼잡구간 출퇴근"),
        (drivers["risky"], "#d64545", "위험 운전자 (정답)"),
    ]
    for mask, color, label in groups:
        d = drivers[mask]
        ax.scatter(d["naive_rate_per100"], d["sir_shrunk"], s=14, c=color, label=label, alpha=0.8)
    ax.axhline(1.0, color="#555", lw=0.8, ls="--")
    ax.set_xlabel("절대 점수: 100회 통과당 급정거")
    ax.set_ylabel("맥락 보정 SIR (관측/기대)")
    ax.set_title("운전자: 절대 점수 vs 맥락 보정")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    tested = cells[cells["exposure"] >= 10].copy()
    is_inc = np.array([k in t_inc for k in zip(tested["segment_id"], tested["bucket"], tested["date"])])
    ax.scatter(tested.loc[~is_inc, "expected"], tested.loc[~is_inc, "count"], s=6, c="#9aa5b1", alpha=0.4, label="일반 셀")
    ax.scatter(tested.loc[is_inc, "expected"], tested.loc[is_inc, "count"], s=40, c="#d64545", label="돌발 (정답)")
    flagged = tested[tested["flagged"]]
    ax.scatter(flagged["expected"], flagged["count"], s=90, facecolors="none", edgecolors="#1f6feb", lw=1.4, label="탐지됨")
    lim = max(tested["expected"].max(), 1)
    ax.plot([0, lim], [0, lim], color="#555", lw=0.8, ls="--")
    ax.set_xlabel("기대 급정거 수 (기준선 × 통과량)")
    ax.set_ylabel("관측 급정거 수")
    ax.set_title("구간 × 날짜 × 시간대 셀")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[2]
    labels = ["구간 돌발", "위험 운전자"]
    x = np.arange(2)
    ax.bar(x - 0.18, [scores["seg_naive"], scores["drv_naive"]], 0.36, color="#9aa5b1", label="절대 기준")
    ax.bar(x + 0.18, [scores["seg_ctx"], scores["drv_ctx"]], 0.36, color="#1f6feb", label="맥락 기준선")
    for i, (a, b) in enumerate([(scores["seg_naive"], scores["seg_ctx"]), (scores["drv_naive"], scores["drv_ctx"])]):
        ax.text(i - 0.18, a + 0.02, f"{a:.2f}", ha="center")
        ax.text(i + 0.18, b + 0.02, f"{b:.2f}", ha="center")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("precision@K")
    ax.set_title("정답 K개 중 상위 K개 적중률")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
