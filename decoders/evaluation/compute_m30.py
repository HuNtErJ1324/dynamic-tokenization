"""
Compute the M30 operating point per (task, lang, boundary).

M30 is the smallest number of dynamic-BPE merges N such that the average
sequence length after N merges is <= 0.70 * the average baseline (N=0)
sequence length on the target evaluation set. It is the merges value the
experiment suite uses for every dynamic_bpe* method, so that every method's
nominal compute budget is matched on a ~30% sequence-length reduction.

Why per (task, lang, boundary)?
  * Different tasks have different prompt distributions, so their merges→seqLen
    curves differ.
  * Korean (KMMLU) tokenizes very differently from English (MMLU).
  * SuperBPE compresses faster than pretokens-only merging, so it hits 30%
    with fewer merges.

Output: data/operating_points/m30_<task>_<lang>_<boundary>.json with the form
    {"task": "mmlu", "lang": "en", "boundary": "superbpe",
     "merges_to_seq_len": {0: 87.3, 1: 87.0, ...},
     "baseline_seq_len": 87.3, "m30": 1234, "m30_seq_len": 61.1,
     "n_examples": 1500}

Usage:
    # Run all 4 combos with defaults:
    python decoders/evaluation/compute_m30.py

    # Single combo:
    python decoders/evaluation/compute_m30.py --task mmlu --lang en \\
        --boundary superbpe --n_examples 1500 --batch_size 32

Requires the hypernet tokenizer (benjamin/zett-hypernetwork-Mistral-7B-v0.1)
to be reachable through huggingface_hub. No GPU needed — this script only
exercises the pure-Python merge simulator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

HOME = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HOME))

from tokenizations.dynamic_bpe import Dynamic_BPE  # noqa: E402


HYPERNET_ID = "benjamin/zett-hypernetwork-Mistral-7B-v0.1"


def _format_mmlu_prompt(question: str, choices: list[str], subject: str) -> str:
    subject = subject.replace("_", " ")
    return (
        f"The following are multiple choice questions (with answers) about {subject}.\n\n"
        f"{question.strip()}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        f"Answer:"
    )


def _format_kmmlu_prompt(question: str, choices: list[str], subject: str) -> str:
    # KMMLU rows expose A/B/C/D as separate columns; mirror MMLU formatting in Korean.
    return (
        f"다음은 {subject}에 관한 객관식 문제입니다.\n\n"
        f"{question.strip()}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        f"정답:"
    )


def _build_mmlu_prompts(n_examples: int) -> Dataset:
    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=0).select(range(min(n_examples, len(ds))))
    prompts = [
        _format_mmlu_prompt(r["question"], r["choices"], r["subject"]) for r in ds
    ]
    return Dataset.from_dict({"prompt": prompts})


def _build_kmmlu_prompts(n_examples: int) -> Dataset:
    # HAERAE-HUB/KMMLU exposes per-subject configs. Concat a handful of subjects so
    # we sample from a diverse Korean prompt distribution.
    subjects = [
        "Accounting", "Biology", "Chemistry", "Civil-Engineering", "Computer-Science",
        "Economics", "Education", "Korean-History", "Law", "Mathematics",
        "Patent", "Public-Safety",
    ]
    rows: list[dict] = []
    per_subject = max(1, n_examples // len(subjects))
    for subject in subjects:
        try:
            sub_ds = load_dataset("HAERAE-HUB/KMMLU", subject, split="test")
        except Exception as e:
            print(f"[warn] could not load KMMLU/{subject}: {e}", flush=True)
            continue
        sub_ds = sub_ds.shuffle(seed=0).select(range(min(per_subject, len(sub_ds))))
        for r in sub_ds:
            rows.append(
                {
                    "prompt": _format_kmmlu_prompt(
                        r["question"], [r["A"], r["B"], r["C"], r["D"]], subject
                    )
                }
            )
    if not rows:
        raise RuntimeError("KMMLU prompt list is empty — check HF cache / network.")
    rows = rows[:n_examples]
    return Dataset.from_dict({"prompt": [r["prompt"] for r in rows]})


def compute_m30(
    task: str,
    lang: str,
    boundary: str,
    n_examples: int = 1500,
    batch_size: int = 32,
    target_ratio: float = 0.70,
    max_merges: int = 5000,
) -> dict:
    """Compute and return the M30 record for one (task, lang, boundary)."""

    if task == "mmlu":
        prompts_ds = _build_mmlu_prompts(n_examples)
    elif task == "kmmlu":
        prompts_ds = _build_kmmlu_prompts(n_examples)
    else:
        raise ValueError(f"Unknown task {task!r}; expected 'mmlu' or 'kmmlu'.")

    tokenizer = AutoTokenizer.from_pretrained(HYPERNET_ID)
    dyn = Dynamic_BPE(tokenizer=tokenizer, tokenizer_boundary=boundary)

    # Use the existing dataset-level simulator. It populates dyn.merges2seqLen with
    # {merges_count -> SUM of seq lengths across all examples in the dataset}.
    dyn.merges2seqLen = {}
    # NOTE: get_merges2seqlen_for_dataset uses an internal max_nr_merges=100000 ceiling.
    # We rely on early-termination + the ratio threshold to cap runtime in practice.
    # If you want a hard ceiling, edit dynamic_bpe.py — out of scope here.
    dyn.get_merges2seqlen_for_dataset(prompts_ds, batch_size=batch_size)

    # merges2seqLen values get divided by len(dataset) at the end of the function,
    # so they're already per-example averages.
    merges_to_avg = {int(k): float(v) for k, v in dyn.merges2seqLen.items()}
    baseline = merges_to_avg[0]
    target = target_ratio * baseline

    # Find M30: smallest N where avg seq len <= target. Iterate in order.
    m30 = None
    for n in sorted(merges_to_avg):
        if merges_to_avg[n] <= target:
            m30 = n
            break

    if m30 is None:
        print(
            f"[warn] {task}/{lang}/{boundary}: did not reach 30% reduction within "
            f"{max(merges_to_avg)} merges. Using the last simulated count as M30.",
            flush=True,
        )
        m30 = max(merges_to_avg)

    return {
        "task": task,
        "lang": lang,
        "boundary": boundary,
        "n_examples": len(prompts_ds),
        "baseline_seq_len": baseline,
        "target_ratio": target_ratio,
        "target_seq_len": target,
        "m30": m30,
        "m30_seq_len": merges_to_avg[m30],
        # JSON keys must be strings; the script that consumes this can re-cast as int.
        "merges_to_seq_len": {str(k): v for k, v in merges_to_avg.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["mmlu", "kmmlu", "all"], default="all")
    parser.add_argument("--lang", default=None, help="Override lang label (e.g. en/ko).")
    parser.add_argument(
        "--boundary",
        choices=["pretokens", "superbpe", "all"],
        default="all",
    )
    parser.add_argument("--n_examples", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=HOME / "data" / "operating_points",
        help="Directory to write m30_<task>_<lang>_<boundary>.json files into.",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = ["mmlu", "kmmlu"] if args.task == "all" else [args.task]
    boundaries = ["pretokens", "superbpe"] if args.boundary == "all" else [args.boundary]

    for task in tasks:
        lang = args.lang or ("en" if task == "mmlu" else "ko")
        for boundary in boundaries:
            print(f"\n=== Computing M30 for {task}/{lang}/{boundary} ===", flush=True)
            record = compute_m30(
                task=task,
                lang=lang,
                boundary=boundary,
                n_examples=args.n_examples,
                batch_size=args.batch_size,
            )
            out_path = out_dir / f"m30_{task}_{lang}_{boundary}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            print(
                f"[done] {task}/{lang}/{boundary}: "
                f"baseline={record['baseline_seq_len']:.2f}, "
                f"M30={record['m30']} merges → {record['m30_seq_len']:.2f} tokens/ex "
                f"(target={record['target_seq_len']:.2f}). "
                f"Wrote {out_path}",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
