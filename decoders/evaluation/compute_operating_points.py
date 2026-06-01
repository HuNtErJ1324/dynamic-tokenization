"""
Compute multiple operating points (M10..M50) per (task, lang, boundary).

Generalizes compute_m30.py: instead of a single 30%-reduction point, run ONE
merge simulation and read off the smallest merge count that reaches each target
reduction in --reductions (default 10,20,30,40,50 percent). The whole curve
comes from a single simulation, so computing five operating points costs the
same as computing one.

Output: data/operating_points/ops_<task>_<lang>_<boundary>.json
    {
      "task": "mmlu", "lang": "en", "boundary": "superbpe",
      "n_examples": 1500,
      "baseline_seq_len": 87.3,
      "operating_points": {
         "10": {"reduction_pct": 10, "target_seq_len": 78.6, "merges": 42,
                "seq_len": 78.1, "achieved_reduction_pct": 10.5, "clamped": false},
         "20": {...}, "30": {...}, "40": {...}, "50": {...}
      },
      "m30": 1234,                      # back-compat: top-level int, like compute_m30.py
      "merges_to_seq_len": {"0": 87.3, "1": 87.0, ...}
    }

run_sweep.sh resolves `merges_source = op:30_pretokens` by reading
operating_points["30"]["merges"] from ops_<task>_<lang>_pretokens.json.

Korean (KMMLU) merging early-exits once the Korean vocabulary is exhausted, so
deep reductions (e.g. 50%) may be unreachable; in that case we clamp to the
largest simulated merge count and set "clamped": true (the cell still runs, just
at the deepest reachable compression).

Usage:
    # All tasks x both boundaries, default reductions (CPU is fine, no GPU):
    python decoders/evaluation/compute_operating_points.py

    # Single combo:
    python decoders/evaluation/compute_operating_points.py --task kmmlu \\
        --boundary superbpe --reductions 10,20,30,40,50 --n_examples 1500

Requires the hypernet tokenizer (benjamin/zett-hypernetwork-Mistral-7B-v0.1)
reachable through huggingface_hub. No GPU needed — pure-Python merge simulator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

HOME = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HOME))

from tokenizations.dynamic_bpe import Dynamic_BPE  # noqa: E402

# Reuse the prompt builders + hypernet id from compute_m30.py (same directory; the
# script's own dir is on sys.path[0] regardless of cwd, so this import is robust).
from compute_m30 import (  # noqa: E402
    HYPERNET_ID,
    _build_kmmlu_prompts,
    _build_mmlu_prompts,
)


def compute_operating_points(
    task: str,
    lang: str,
    boundary: str,
    reductions: list[int],
    n_examples: int = 1500,
    batch_size: int = 32,
) -> dict:
    """Run one merge simulation and read off an operating point per reduction %."""
    if task == "mmlu":
        prompts_ds = _build_mmlu_prompts(n_examples)
    elif task == "kmmlu":
        prompts_ds = _build_kmmlu_prompts(n_examples)
    else:
        raise ValueError(f"Unknown task {task!r}; expected 'mmlu' or 'kmmlu'.")

    tokenizer = AutoTokenizer.from_pretrained(HYPERNET_ID)
    dyn = Dynamic_BPE(tokenizer=tokenizer, tokenizer_boundary=boundary)

    # Single simulation: populates dyn.merges2seqLen with per-example avg seq len
    # at every merge count (it divides by len(dataset) internally).
    dyn.merges2seqLen = {}
    dyn.get_merges2seqlen_for_dataset(prompts_ds, batch_size=batch_size)

    merges_to_avg = {int(k): float(v) for k, v in dyn.merges2seqLen.items()}
    baseline = merges_to_avg[0]

    # Compression saturates once no mergeable pairs remain — beyond that, more
    # "merges" shorten nothing. Find the FEWEST merges that reach the minimum
    # achievable seq len and clamp unreachable targets to THAT, not the simulator's
    # ceiling (otherwise we'd emit absurd, un-runnable counts like 99999).
    min_seq = min(merges_to_avg.values())
    sat_merges = min(n for n in merges_to_avg if merges_to_avg[n] <= min_seq + 1e-9)
    max_achievable_reduction = round(100.0 * (1.0 - min_seq / baseline), 2)

    def _find_merges(target_seq_len: float) -> tuple[int, float, bool]:
        """Smallest N whose avg seq len <= target; clamp to the saturation point."""
        for n in sorted(merges_to_avg):
            if merges_to_avg[n] <= target_seq_len:
                return n, merges_to_avg[n], False
        return sat_merges, merges_to_avg[sat_merges], True

    operating_points: dict[str, dict] = {}
    for pct in reductions:
        ratio = 1.0 - pct / 100.0
        target = ratio * baseline
        merges, seq_len, clamped = _find_merges(target)
        if clamped:
            print(
                f"[warn] {task}/{lang}/{boundary}: {pct}% reduction unreachable "
                f"(max achievable ~{max_achievable_reduction}%); clamping to the "
                f"saturation point ({sat_merges} merges). Consider dropping this "
                f"target from --reductions for this boundary.",
                flush=True,
            )
        operating_points[str(pct)] = {
            "reduction_pct": pct,
            "target_seq_len": target,
            "merges": merges,
            "seq_len": seq_len,
            "achieved_reduction_pct": round(100.0 * (1.0 - seq_len / baseline), 2),
            "clamped": clamped,
        }

    record = {
        "task": task,
        "lang": lang,
        "boundary": boundary,
        "n_examples": len(prompts_ds),
        "baseline_seq_len": baseline,
        "max_achievable_reduction_pct": max_achievable_reduction,
        "saturation_merges": sat_merges,
        "operating_points": operating_points,
        # JSON keys must be strings; consumers re-cast merge counts as int.
        "merges_to_seq_len": {str(k): v for k, v in merges_to_avg.items()},
    }
    # Back-compat with compute_m30.py / run_suite.sh's read_m30: expose a top-level int.
    if "30" in operating_points:
        record["m30"] = operating_points["30"]["merges"]
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=["mmlu", "kmmlu", "all"], default="all")
    parser.add_argument("--lang", default=None, help="Override lang label (e.g. en/ko).")
    parser.add_argument(
        "--boundary",
        choices=["pretokens", "word", "word_hyphen", "sentence", "superbpe", "all"],
        default="all",
        help="Merge boundary. 'all' = the two primary boundaries (pretokens, superbpe). "
        "Pass word/word_hyphen/sentence explicitly for the S4 boundary ablation.",
    )
    parser.add_argument(
        "--reductions",
        default="10,20,30,40,50",
        help="Comma-separated reduction percentages to compute operating points for.",
    )
    parser.add_argument("--n_examples", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=HOME / "data" / "operating_points",
        help="Directory to write ops_<task>_<lang>_<boundary>.json files into.",
    )
    args = parser.parse_args()

    reductions = [int(x) for x in args.reductions.split(",") if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tasks = ["mmlu", "kmmlu"] if args.task == "all" else [args.task]
    boundaries = (
        ["pretokens", "superbpe"] if args.boundary == "all" else [args.boundary]
    )

    for task in tasks:
        lang = args.lang or ("en" if task == "mmlu" else "ko")
        for boundary in boundaries:
            print(
                f"\n=== Operating points for {task}/{lang}/{boundary} "
                f"(reductions={reductions}) ===",
                flush=True,
            )
            record = compute_operating_points(
                task=task,
                lang=lang,
                boundary=boundary,
                reductions=reductions,
                n_examples=args.n_examples,
                batch_size=args.batch_size,
            )
            out_path = args.out_dir / f"ops_{task}_{lang}_{boundary}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            pts = ", ".join(
                f"M{p}={record['operating_points'][str(p)]['merges']}"
                f"({record['operating_points'][str(p)]['achieved_reduction_pct']}%)"
                for p in reductions
            )
            print(
                f"[done] {task}/{lang}/{boundary}: "
                f"baseline={record['baseline_seq_len']:.2f} tok/ex; "
                f"max reduction ~{record['max_achievable_reduction_pct']}% "
                f"@ {record['saturation_merges']} merges; {pts}. "
                f"Wrote {out_path}",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
