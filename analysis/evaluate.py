"""Scores the detectors on a labelled dataset (e.g. SUMO output from sumo_sim/generate.py).

    python evaluate.py out/sumo

Needs in the folder: traversals.csv, truth_incidents.csv (segment_id, date, bucket),
truth_drivers.csv (driver_id, label: normal / drowsy / aggressive).
Rows with driver_id == "bg" (anonymous background traffic) feed the road baseline but
are not scored as drivers.

Compares, with K = number of true anomalies:
  segments: raw count rank, raw rate rank (context-free)  vs  Poisson p-value vs baseline
  drivers : hard brakes per 100 traversals (TMAP-style)    vs  context-adjusted SIR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline import (add_context, detect_driver_anomalies, detect_segment_anomalies, detect_window_spikes,
                      slot_label)

LABEL_COLOR = {"normal": "#9aa5b1", "drowsy": "#7c4dff", "aggressive": "#d64545"}
LABEL_KO = {"normal": "정상", "drowsy": "졸음", "aggressive": "공격적"}


def _key(d):
    return set(zip(d["segment_id"], d["bucket"], d["date"]))


def incident_cells(truth_inc: pd.DataFrame, slot_minutes: int | None) -> dict[tuple, int]:
    """(segment, bucket, date) -> incident index. In slot mode an incident covers every
    slot its time window overlaps; in band mode it is the band cell of its start."""
    out = {}
    for i, r in enumerate(truth_inc.itertuples()):
        day_type = "weekend" if pd.Timestamp(r.date).dayofweek >= 5 else "weekday"
        if not slot_minutes:
            out[(r.segment_id, r.bucket, r.date)] = i
            continue
        for m in range(int(r.start_s) // 60, (int(r.end_s) - 1) // 60 + 1, slot_minutes):
            label = slot_label([m // 60], [m % 60], slot_minutes).iloc[0]
            out[(r.segment_id, f"{day_type}_{label}", r.date)] = i
        last = (int(r.end_s) - 1) // 60
        out[(r.segment_id, f"{day_type}_{slot_label([last // 60], [last % 60], slot_minutes).iloc[0]}", r.date)] = i
    return out


def effective_incidents(folder: Path, truth_inc: pd.DataFrame, n_dates: int) -> set[int]:
    """Indices of planted incidents that actually changed driving.

    SUMO only: compares hard brakes in the incident's 20 minutes with the same 20 minutes
    on other days (+5 or more and at least 2x). Incidents on congested or empty roads
    barely affect anyone and cannot be detected from driving data at all.
    """
    if not (folder / "events.csv").exists() or "start_s" not in truth_inc.columns:
        return set(range(len(truth_inc)))
    ev = pd.read_csv(folder / "events.csv", usecols=["segment_id", "date", "t"])
    out = set()
    for i, r in enumerate(truth_inc.itertuples()):
        win = ev[(ev["segment_id"] == r.segment_id) & (ev["t"] >= r.start_s) & (ev["t"] < r.end_s)]
        during = (win["date"] == r.date).sum()
        usual = (win["date"] != r.date).sum() / max(n_dates - 1, 1)
        if during - usual >= 5 and during >= 2 * usual:
            out.add(i)
    return out


def evaluate(folder: Path, slot_minutes: int | None = None, min_driver_traversals: int = 50):
    trav = pd.read_csv(folder / "traversals.csv")
    truth_inc = pd.read_csv(folder / "truth_incidents.csv")
    truth_drv = pd.read_csv(folder / "truth_drivers.csv")

    df = add_context(trav, slot_minutes)
    cells = detect_segment_anomalies(df)
    scored = df[df["driver_id"] != "bg"]
    drivers = detect_driver_anomalies(scored, cells).merge(truth_drv[["driver_id", "label"]], on="driver_id")
    drivers = drivers[drivers["traversals"] >= min_driver_traversals].reset_index(drop=True)

    # Route context: how brake-prone a driver's roads/times are, independent of the driver.
    drivers["route_context"] = drivers["expected"] / drivers["traversals"] * 100
    drivers["congested_route"] = drivers["route_context"] >= drivers["route_context"].quantile(2 / 3)
    drivers["risky"] = drivers["label"] != "normal"

    # ---- segments (scored per incident: found if any of its cells is picked)
    inc_of = incident_cells(truth_inc, slot_minutes)
    t_inc = set(inc_of)
    k_inc = max(len(truth_inc), 1)
    tested = cells[cells["exposure"] >= 10].copy()
    tested["rate"] = tested["count"] / tested["exposure"]

    def found(picked) -> float:
        return len({inc_of[c] for c in _key(picked) if c in inc_of}) / k_inc

    effective = effective_incidents(folder, truth_inc, df["date"].nunique())
    k_eff = max(len(effective), 1)

    def found_eff(picked, index) -> float:
        return len({index[c] for c in _key(picked) if c in index} & effective) / k_eff

    flagged = _key(cells[cells["flagged"]])
    seg = {
        "count": found(tested.nlargest(k_inc, "count")),
        "rate": found(tested.nlargest(k_inc, "rate")),
        "context": found(tested.nsmallest(k_inc, "p_value")),
        "flag_n": len(flagged),
        "flag_p": len(flagged & t_inc) / len(flagged) if flagged else 0.0,
        "flag_r": found(cells[cells["flagged"]]),
        "k": len(truth_inc),
        "k_eff": len(effective),
        "context_eff": found_eff(tested.nsmallest(k_inc, "p_value"), inc_of),
        "count_eff": found_eff(tested.nlargest(k_inc, "count"), inc_of),
        "mode": f"{slot_minutes}분 슬롯" if slot_minutes else "시간대 밴드",
    }

    # ---- sliding-window scan inside each band cell (needs entry minutes)
    if "minute" in trav.columns:
        band_df = add_context(trav) if slot_minutes else df
        band_cells = detect_segment_anomalies(band_df) if slot_minutes else cells
        scan = detect_window_spikes(band_df, band_cells)
        inc_band = incident_cells(truth_inc, None)
        found_band = lambda picked: len({inc_band[c] for c in _key(picked) if c in inc_band}) / k_inc
        flagged_scan = _key(scan[scan["flagged"]])
        seg.update({
            "scan": found_band(scan[scan["exposure"] >= 10].nsmallest(k_inc, "p_value")),
            "scan_eff": found_eff(scan[scan["exposure"] >= 10].nsmallest(k_inc, "p_value"), inc_band),
            "scan_flag_n": len(flagged_scan),
            "scan_flag_p": len(flagged_scan & set(inc_band)) / len(flagged_scan) if flagged_scan else 0.0,
            "scan_flag_r": found_band(scan[scan["flagged"]]),
        })
        scan.to_csv(folder / "segment_window_scan.csv", index=False)

    # ---- drivers
    k_drv = int(drivers["risky"].sum())
    top_naive = drivers.nlargest(k_drv, "naive_rate_per100")
    top_ctx = drivers.nlargest(k_drv, "sir_shrunk")
    flag = drivers[drivers["flagged"]]
    drv = {
        "k": k_drv,
        "naive": top_naive["risky"].mean(),
        "context": top_ctx["risky"].mean(),
        "naive_fp_congested": top_naive.loc[~top_naive["risky"], "congested_route"].mean(),
        "ctx_fp_congested": top_ctx.loc[~top_ctx["risky"], "congested_route"].mean(),
        "naive_fp": int((~top_naive["risky"]).sum()),
        "ctx_fp": int((~top_ctx["risky"]).sum()),
        "flag_n": len(flag),
        "flag_p": flag["risky"].mean() if len(flag) else 0.0,
        "flag_r": flag["risky"].sum() / k_drv if k_drv else 0.0,
        "recall_by_label": {lab: (top_ctx["label"] == lab).sum() / max((drivers["label"] == lab).sum(), 1)
                            for lab in ("drowsy", "aggressive")},
        "naive_recall_by_label": {lab: (top_naive["label"] == lab).sum() / max((drivers["label"] == lab).sum(), 1)
                                  for lab in ("drowsy", "aggressive")},
        "corr_naive_route": drivers.loc[~drivers["risky"], ["naive_rate_per100", "route_context"]].corr().iloc[0, 1],
        "corr_sir_route": drivers.loc[~drivers["risky"], ["sir", "route_context"]].corr().iloc[0, 1],
    }
    return df, cells, drivers, t_inc, seg, drv


def report(df, drivers, seg, drv) -> str:
    rates = drivers.groupby("label")["naive_rate_per100"].mean()
    return "\n".join([
        "# 급정거 이상탐지 평가 (SUMO)",
        "",
        f"- 통과 {len(df):,}건 (배경 교통 포함), 급정거 {int(df['hard_brakes'].sum()):,}회, "
        f"구간 {df['segment_id'].nunique()}개, {df['date'].nunique()}일",
        f"- 평가 운전자 {len(drivers)}명: " + ", ".join(
            f"{LABEL_KO[l]} {int((drivers['label'] == l).sum())}명 (평균 {rates.get(l, float('nan')):.1f}/100회)"
            for l in ("normal", "drowsy", "aggressive")),
        f"- 심어둔 구간 돌발: {seg['k']}건, 시간 단위: {seg['mode']}",
        "",
        "## 구간 이상 (구간 × 날짜 × 시간 단위): 상위 K개 셀로 찾아낸 돌발 비율",
        "",
        "| 방법 | 적중률 |",
        "|---|---|",
        f"| 절대: 급정거 횟수 순 | {seg['count']:.2f} |",
        f"| 절대: 급정거율 순 | {seg['rate']:.2f} |",
        f"| 맥락 기준선 (p값 순) | {seg['context']:.2f} |",
        f"| 맥락 기준선 자체 플래그 | precision {seg['flag_p']:.2f}, recall {seg['flag_r']:.2f} ({seg['flag_n']}건) |",
        *([f"| 맥락 스캔 (10~30분 창, p값 순) | {seg['scan']:.2f} |",
           f"| 맥락 스캔 자체 플래그 | precision {seg['scan_flag_p']:.2f}, recall {seg['scan_flag_r']:.2f} "
           f"({seg['scan_flag_n']}건) |"] if "scan" in seg else []),
        "",
        f"- 실제로 급정거를 늘린 돌발 {seg['k_eff']}건 기준 적중률: 절대(횟수 순) {seg['count_eff']:.2f}, "
        f"맥락 기준선 {seg['context_eff']:.2f}"
        + (f", 맥락 스캔 {seg['scan_eff']:.2f}" if "scan_eff" in seg else "")
        + "  (나머지는 정체·한산한 시간이라 돌발이 운전에 거의 영향을 주지 않음)",
        "",
        f"## 위험 운전자 (졸음 + 공격적, K = {drv['k']})",
        "",
        "| 방법 | precision@K | 졸음 recall | 공격적 recall | 오탐 중 혼잡경로 비율 |",
        "|---|---|---|---|---|",
        f"| 절대 점수 (100회당 급정거) | {drv['naive']:.2f} | {drv['naive_recall_by_label']['drowsy']:.2f} | "
        f"{drv['naive_recall_by_label']['aggressive']:.2f} | {drv['naive_fp_congested']:.0%} ({drv['naive_fp']}명 중) |",
        f"| 맥락 보정 SIR | {drv['context']:.2f} | {drv['recall_by_label']['drowsy']:.2f} | "
        f"{drv['recall_by_label']['aggressive']:.2f} | {drv['ctx_fp_congested']:.0%} ({drv['ctx_fp']}명 중) |",
        "",
        f"- 맥락 보정 자체 플래그: precision {drv['flag_p']:.2f}, recall {drv['flag_r']:.2f} ({drv['flag_n']}명)",
        f"- 정상 운전자에서 점수와 경로 혼잡도의 상관: 절대 점수 {drv['corr_naive_route']:.2f}, "
        f"맥락 SIR {drv['corr_sir_route']:.2f}  (0에 가까울수록 길 탓이 점수에 덜 섞임)",
        "",
    ])


def plot(drivers, cells, t_inc, seg, drv, path: Path):
    for font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if any(font in f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    for lab in ("normal", "drowsy", "aggressive"):
        d = drivers[drivers["label"] == lab]
        for cong, marker in ((False, "o"), (True, "^")):
            dd = d[d["congested_route"] == cong]
            ax.scatter(dd["naive_rate_per100"], dd["sir_shrunk"], s=18, marker=marker, c=LABEL_COLOR[lab],
                       alpha=0.8, label=f"{LABEL_KO[lab]}{' · 혼잡경로' if cong else ''}")
    ax.axhline(1.0, color="#555", lw=0.8, ls="--")
    ax.set_xlabel("절대 점수: 100회 통과당 급정거")
    ax.set_ylabel("맥락 보정 SIR (관측/기대)")
    ax.set_title("운전자: 절대 점수 vs 맥락 보정")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    tested = cells[cells["exposure"] >= 10]
    is_inc = np.array([k in t_inc for k in zip(tested["segment_id"], tested["bucket"], tested["date"])], dtype=bool)
    ax.scatter(tested.loc[~is_inc, "expected"], tested.loc[~is_inc, "count"], s=5, c="#9aa5b1", alpha=0.35, label="일반 셀")
    ax.scatter(tested.loc[is_inc, "expected"], tested.loc[is_inc, "count"], s=40, c="#d64545", label="돌발 (정답)")
    f = tested[tested["flagged"]]
    ax.scatter(f["expected"], f["count"], s=90, facecolors="none", edgecolors="#1f6feb", lw=1.4, label="탐지됨")
    lim = max(tested["expected"].max(), 1)
    ax.plot([0, lim], [0, lim], color="#555", lw=0.8, ls="--")
    ax.set_xlabel("기대 급정거 수 (기준선 × 통과량)")
    ax.set_ylabel("관측 급정거 수")
    ax.set_title("구간 × 날짜 × 시간대 셀")
    ax.legend(fontsize=9)

    ax = axes[2]
    x = np.arange(2)
    naive = [max(seg["count"], seg["rate"]), drv["naive"]]
    ctx = [seg["context"], drv["context"]]
    ax.bar(x - 0.18, naive, 0.36, color="#9aa5b1", label="절대 기준 (더 나은 쪽)")
    ax.bar(x + 0.18, ctx, 0.36, color="#1f6feb", label="맥락 기준선")
    for i in range(2):
        ax.text(i - 0.18, naive[i] + 0.02, f"{naive[i]:.2f}", ha="center")
        ax.text(i + 0.18, ctx[i] + 0.02, f"{ctx[i]:.2f}", ha="center")
    ax.set_xticks(x, ["구간 돌발", "위험 운전자"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("precision@K")
    ax.set_title("정답 K개 중 상위 K개 적중률")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--slot-minutes", type=int, default=None,
                    help="e.g. 30: baseline per 30-min slot (shrunk toward its time band) instead of per band")
    args = ap.parse_args()
    df, cells, drivers, t_inc, seg, drv = evaluate(args.folder, args.slot_minutes)
    text = report(df, drivers, seg, drv)
    tag = f"_slot{args.slot_minutes}" if args.slot_minutes else ""
    (args.folder / f"summary{tag}.md").write_text(text, encoding="utf-8")
    drivers.to_csv(args.folder / f"driver_scores{tag}.csv", index=False)
    cells[cells["flagged"]].to_csv(args.folder / f"segment_anomalies{tag}.csv", index=False)
    plot(drivers, cells, t_inc, seg, drv, args.folder / f"results{tag}.png")
    print(text)


if __name__ == "__main__":
    main()
