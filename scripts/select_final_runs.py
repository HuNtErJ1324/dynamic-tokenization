"""
Pick the configs worth re-running on the FULL test set, from the subsample sweep
results (results/suite_results.csv), and emit sweeps/final.tsv (max_examples=0).

Per task it selects:
  * baselines           — plain, original_tk_hypernet (reference points)
  * Pareto-optimal       — configs non-dominated in (sequence-reduction%, accuracy)
  * best-threshold       — the max-accuracy entropy threshold per (exp_type, boundary)

Random-split (S8 control) rows are excluded — the subsample result already makes that
point; no need to spend full-set budget on it. The output is a STARTING recommendation:
eyeball the printed table (and split_rate in the CSV) before submitting.

Usage:
    python scripts/select_final_runs.py
    python scripts/select_final_runs.py --csv results/suite_results.csv --out sweeps/final.tsv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _key(r):
    return (
        r.get("task", ""), r.get("exp_type", ""), str(r.get("merges", "")),
        r.get("boundary", ""), str(r.get("transition_point", "0")),
        str(r.get("entropy_threshold", "")), r.get("split_strategy", "entropy"),
    )


def pareto(rows):
    """Non-dominated rows maximizing BOTH sequence-reduction% and accuracy."""
    pts = [(r, _f(r.get("seq_reduction_pct")), _f(r.get("accuracy"))) for r in rows]
    pts = [(r, x, y) for r, x, y in pts if x is not None and y is not None]
    keep = []
    for r, x, y in pts:
        if not any((x2 >= x and y2 >= y and (x2 > x or y2 > y)) for _, x2, y2 in pts):
            keep.append(r)
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("results/suite_results.csv"))
    ap.add_argument("--out", type=Path, default=Path("sweeps/final.tsv"))
    args = ap.parse_args()

    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    rows = [r for r in rows if _f(r.get("accuracy")) is not None]          # drop failed/missing
    rows = [r for r in rows if (r.get("split_strategy") or "entropy") != "random"]  # control, not final

    chosen, reason = {}, {}
    by_task = defaultdict(list)
    for r in rows:
        by_task[r.get("task", "")].append(r)

    for task, trows in by_task.items():
        for r in trows:  # baselines
            if r.get("exp_type") in ("plain", "original_tk_hypernet") and str(r.get("merges", "")) in ("0", ""):
                chosen[_key(r)] = r; reason[_key(r)] = "baseline"
        for r in pareto(trows):  # accuracy-vs-compression frontier
            chosen.setdefault(_key(r), r); reason.setdefault(_key(r), "pareto")
        best = {}  # best entropy threshold per (exp_type, boundary)
        for r in trows:
            if r.get("exp_type") in ("entropy_split", "dynamic_bpe_entropy_split"):
                k = (r.get("exp_type"), r.get("boundary"))
                if k not in best or (_f(r.get("accuracy")) or 0) > (_f(best[k].get("accuracy")) or 0):
                    best[k] = r
        for r in best.values():
            chosen.setdefault(_key(r), r); reason.setdefault(_key(r), "best-threshold")

    HEADER = ["tag", "task", "exp_type", "merges_source", "boundary", "transition_point",
              "threshold", "batch_size", "max_examples", "split_strategy"]
    lines = ["\t".join(HEADER)]
    out_rows = sorted(chosen.values(), key=lambda r: (r.get("task"), -(_f(r.get("accuracy")) or 0)))

    print(f"\n{'task':6} {'exp_type':28} {'merges':>7} {'boundary':11} {'thr':>5} "
          f"{'redux%':>7} {'acc':>7}  why")
    print("-" * 86)
    for i, r in enumerate(out_rows):
        is_split = "split" in (r.get("exp_type") or "")
        thr = r.get("entropy_threshold", "3.0") if is_split else "-"
        tag = f"FIN_{r.get('task')}_{(r.get('exp_type') or '')[:8]}_{i}"
        lines.append("\t".join([
            tag, r.get("task", ""), r.get("exp_type", ""), str(r.get("merges") or 0),
            r.get("boundary") or "pretokens", str(r.get("transition_point") or 0),
            str(thr), "4", "0", r.get("split_strategy") or "entropy",
        ]))
        print(f"{r.get('task',''):6} {r.get('exp_type',''):28} {str(r.get('merges','')):>7} "
              f"{(r.get('boundary') or ''):11} {str(thr):>5} "
              f"{str(r.get('seq_reduction_pct','')):>7} {str(r.get('accuracy','')):>7}  {reason[_key(r)]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"\nWrote {len(out_rows)} final-run configs -> {args.out}")
    print(f"Submit full-set runs with:  WALLTIME=12:00:00 bash scripts/run_sweep.sh {args.out} --submit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
