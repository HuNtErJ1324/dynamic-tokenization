"""
Plot the sweep results produced by scripts/aggregate_results.py.

Reads results/suite_results.csv and writes figures into results/figures/:
  - pareto_<task>.png          accuracy vs sequence-reduction% (one line per method) + baselines  [S1]
  - threshold_sweep_<task>.png accuracy (solid) & split_rate (dashed) vs entropy threshold        [S2]
  - parity_en_vs_ko.png        MMLU(en) vs KMMLU(ko) accuracy per method at M30                    [S1]
  - throughput_<task>.png      throughput (ex/s) vs sequence-reduction%                            [S1]

Each figure is skipped (with a note) when the underlying rows aren't present, so
this is safe to run after a partial sweep.

Usage:
    python scripts/plot_pareto.py
    python scripts/plot_pareto.py --csv results/suite_results.csv --outdir results/figures
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed on the cluster
import matplotlib.pyplot as plt  # noqa: E402


def _f(x):
    """float-or-None coercion for possibly-empty CSV cells."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def series_label(row: dict) -> str:
    e = row.get("exp_type", "")
    b = row.get("boundary", "")
    return {
        "plain": "plain (baseline)",
        "original_tk_hypernet": "orig-tok + HN",
        "entropy_split": "entropy-split",
        "dynamic_bpe": f"dyn-bpe ({b})",
        "dynamic_bpe_entropy_split": f"merge+split ({b})",
    }.get(e, f"{e} ({b})")


def plot_pareto(rows, outdir):
    written = []
    for task in sorted({r["task"] for r in rows if r.get("sweep") == "S1_pareto"}):
        trows = [r for r in rows if r.get("task") == task and r.get("sweep") == "S1_pareto"]
        series = defaultdict(list)
        baselines = {}
        for r in trows:
            acc = _f(r.get("accuracy"))
            if acc is None:
                continue
            if r.get("exp_type") in ("plain", "original_tk_hypernet"):
                baselines[series_label(r)] = acc
                continue
            red = _f(r.get("seq_reduction_pct"))
            if red is None:
                continue
            series[series_label(r)].append((red, acc))
        if not series and not baselines:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        for lab, pts in sorted(series.items()):
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=lab)
        for lab, acc in sorted(baselines.items()):
            ax.axhline(acc, ls="--", alpha=0.6, label=lab)
        ax.set_xlabel("sequence reduction (%)")
        ax.set_ylabel("accuracy")
        ax.set_title(f"Accuracy vs compression — {task}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        out = outdir / f"pareto_{task}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written


def plot_threshold(rows, outdir):
    written = []
    for task in sorted({r["task"] for r in rows if r.get("sweep") == "S2_threshold"}):
        trows = [r for r in rows if r.get("task") == task and r.get("sweep") == "S2_threshold"]
        acc_series = defaultdict(list)
        split_series = defaultdict(list)
        for r in trows:
            thr = _f(r.get("entropy_threshold"))
            if thr is None or thr > 50:  # drop the no-split sentinel (e.g. 100) from the curve
                continue
            lab = series_label(r)
            acc = _f(r.get("accuracy"))
            sr = _f(r.get("split_rate"))
            if acc is not None:
                acc_series[lab].append((thr, acc))
            if sr is not None:
                split_series[lab].append((thr, sr))
        if not acc_series:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        ax2 = ax.twinx()
        for lab, pts in sorted(acc_series.items()):
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"{lab} acc")
        for lab, pts in sorted(split_series.items()):
            pts.sort()
            ax2.plot([p[0] for p in pts], [p[1] for p in pts], ls="--", marker="x", alpha=0.6)
        ax.set_xlabel("entropy threshold")
        ax.set_ylabel("accuracy (solid)")
        ax2.set_ylabel("split_rate (dashed)")
        ax.set_title(f"Entropy-threshold sweep — {task}")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        out = outdir / f"threshold_sweep_{task}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written


def plot_parity(rows, outdir):
    # Accuracy at M30 (op:30) per method series, MMLU(en) vs KMMLU(ko).
    sel = [r for r in rows if r.get("sweep") == "S1_pareto" and (r.get("op") or "").startswith("op:30")]
    by_series = defaultdict(dict)  # label -> {task: acc}
    for r in sel:
        acc = _f(r.get("accuracy"))
        if acc is not None:
            by_series[series_label(r)][r.get("task")] = acc
    labels = sorted(s for s in by_series if {"mmlu", "kmmlu"} & set(by_series[s]))
    if not labels:
        return []
    en = [by_series[s].get("mmlu", float("nan")) for s in labels]
    ko = [by_series[s].get("kmmlu", float("nan")) for s in labels]
    x = list(range(len(labels)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.7), 5))
    ax.bar([i - w / 2 for i in x], en, w, label="MMLU (en)")
    ax.bar([i + w / 2 for i in x], ko, w, label="KMMLU (ko)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("accuracy")
    ax.set_title("English vs Korean @ M30 (30% reduction)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    out = outdir / "parity_en_vs_ko.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [out]


def plot_throughput(rows, outdir):
    written = []
    for task in sorted({r["task"] for r in rows if r.get("sweep") == "S1_pareto"}):
        trows = [r for r in rows if r.get("task") == task and r.get("sweep") == "S1_pareto"]
        series = defaultdict(list)
        for r in trows:
            thr = _f(r.get("throughput_ex_per_s"))
            red = _f(r.get("seq_reduction_pct"))
            if thr is None or red is None:
                continue
            series[series_label(r)].append((red, thr))
        if not series:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        for lab, pts in sorted(series.items()):
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=lab)
        ax.set_xlabel("sequence reduction (%)")
        ax.set_ylabel("throughput (ex/s)")
        ax.set_title(f"Throughput vs compression — {task}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        out = outdir / f"throughput_{task}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=repo_root / "results" / "suite_results.csv")
    parser.add_argument("--outdir", type=Path, default=repo_root / "results" / "figures")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: {args.csv} not found. Run scripts/aggregate_results.py first.")
        return 1

    rows = load_rows(args.csv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    written = []
    for fn in (plot_pareto, plot_threshold, plot_parity, plot_throughput):
        try:
            written.extend(fn(rows, args.outdir))
        except Exception as e:  # never let one bad plot kill the rest
            print(f"[warn] {fn.__name__} failed: {type(e).__name__}: {e}", flush=True)

    if written:
        for p in written:
            print(f"wrote {p}")
    else:
        print("No figures written — no matching rows yet (run/aggregate a sweep first).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
