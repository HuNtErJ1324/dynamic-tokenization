"""
Aggregate slurm logs from a run of scripts/run_suite.sh into a single results
table. Reads the manifest at logs/suite_manifest.tsv (job_id ↔ method ↔ config)
and pairs every job with its matching slurm log under logs/, parsing the
stdout for accuracy, latency, and merged-token-length stats.

Outputs:
  - results/suite_results.csv     — one row per (task, method)
  - results/suite_results.md      — same content as a Markdown table for the report

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --manifest path/to/manifest.tsv --logs path/to/logs/
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Optional


# ── regexes ──────────────────────────────────────────────────────────────────

# `Overall MMLU accuracy: 0.6783 (1234/1820)` or `Overall KMMLU accuracy: 0.5421 (...)`
RE_ACCURACY = re.compile(
    r"Overall (?P<dataset>\w+) accuracy:\s*(?P<acc>[0-9.]+)\s*\((?P<correct>\d+)/(?P<seen>\d+)\)"
)

# `[latency] total_wall_time=1234.56s   examples=1820   throughput=1.47 ex/s   per_example=678.90 ms`
RE_LATENCY = re.compile(
    r"\[latency\]\s+total_wall_time=(?P<wall>[0-9.]+)s\s+examples=(?P<n>\d+)\s+"
    r"throughput=(?P<thr>[0-9.]+)\s*ex/s\s+per_example=(?P<ms_per_ex>[0-9.]+)\s*ms"
)

# `[mmlu] merged-token byte length: n=12345, mean=4.32, p95=8, max=21, frac>7=0.1234`
# `[kmmlu_pre_split] merged-token byte length: ...`
RE_TOK_LEN = re.compile(
    r"\[(?P<label>[\w_]+)\]\s+merged-token byte length:\s+"
    r"n=(?P<n>\d+),\s*mean=(?P<mean>[0-9.]+),\s*p95=(?P<p95>\d+),\s*"
    r"max=(?P<max>\d+),\s*frac>(?P<hn_maxlen>\d+)=(?P<frac>[0-9.]+)"
)

# `[dyn_bpe_entropy_split] entropy_threshold=3.00  split_rate=0.0421  (1234/29345 merged tokens re-split)`
RE_SPLIT_RATE = re.compile(
    r"\[dyn_bpe_entropy_split\]\s+entropy_threshold=(?P<thr>[0-9.]+)\s+"
    r"split_rate=(?P<rate>[0-9.]+)\s+\((?P<num>\d+)/(?P<den>\d+)"
)


def _read_manifest(path: Path) -> list[dict]:
    """Read the TSV manifest produced by run_suite.sh --submit."""
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {path}. "
            "Run scripts/run_suite.sh --submit first, or pass --manifest explicitly."
        )
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def _find_log(logs_dir: Path, job_id: str) -> Optional[Path]:
    """Find the slurm log for a job id. Looks for *-{job_id}.out under logs_dir."""
    candidates = list(logs_dir.glob(f"*-{job_id}.out"))
    if not candidates:
        # also try *_{job_id}.out / {job_id}.out as fallbacks
        candidates = list(logs_dir.glob(f"*{job_id}*.out"))
    if not candidates:
        return None
    # If multiple matches (shouldn't happen), return the most recent.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_log(path: Path) -> dict:
    """Pull accuracy / latency / token-length stats out of one slurm stdout file."""
    text = path.read_text(encoding="utf-8", errors="replace")

    out: dict = {}

    if m := RE_ACCURACY.search(text):
        out["accuracy"] = float(m["acc"])
        out["correct"] = int(m["correct"])
        out["seen"] = int(m["seen"])

    if m := RE_LATENCY.search(text):
        out["wall_time_s"] = float(m["wall"])
        out["throughput_ex_per_s"] = float(m["thr"])
        out["ms_per_example"] = float(m["ms_per_ex"])

    # Capture every merged-token-length line. Methods like dyn_bpe_entropy_split
    # emit both *_pre_split and *_post_split versions.
    for m in RE_TOK_LEN.finditer(text):
        label = m["label"]
        out[f"toklen_{label}_n"] = int(m["n"])
        out[f"toklen_{label}_mean"] = float(m["mean"])
        out[f"toklen_{label}_p95"] = int(m["p95"])
        out[f"toklen_{label}_max"] = int(m["max"])
        out[f"toklen_{label}_frac_gt_hn"] = float(m["frac"])

    if m := RE_SPLIT_RATE.search(text):
        out["split_rate"] = float(m["rate"])
        out["split_count"] = int(m["num"])
        out["split_total"] = int(m["den"])

    return out


def aggregate(manifest_path: Path, logs_dir: Path) -> list[dict]:
    manifest = _read_manifest(manifest_path)
    rows = []
    for entry in manifest:
        job_id = entry.get("job_id", "").strip()
        log_path = _find_log(logs_dir, job_id) if job_id else None
        merged = dict(entry)
        merged["log_path"] = str(log_path) if log_path else ""
        if log_path is not None:
            merged.update(_parse_log(log_path))
        else:
            print(f"[warn] no log for job_id={job_id} method={entry.get('method')} task={entry.get('task')}", flush=True)
        rows.append(merged)
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    # Union all keys to a stable header (manifest columns first, then numeric metrics).
    base_cols = [
        "submitted_at", "job_id", "method", "task", "exp_type", "merges",
        "boundary", "transition_point", "entropy_threshold",
        "accuracy", "correct", "seen",
        "wall_time_s", "throughput_ex_per_s", "ms_per_example",
        "split_rate", "split_count", "split_total",
    ]
    extra_cols = sorted({k for r in rows for k in r if k not in base_cols and k != "log_path"})
    cols = base_cols + extra_cols + ["log_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def write_markdown(rows: list[dict], out_path: Path) -> None:
    """Tighter Markdown table for the report — just the columns reviewers want."""
    cols = [
        ("Method", "method"),
        ("Task", "task"),
        ("exp_type", "exp_type"),
        ("Boundary", "boundary"),
        ("Merges", "merges"),
        ("Threshold", "entropy_threshold"),
        ("TP", "transition_point"),
        ("Accuracy", "accuracy"),
        ("ms/ex", "ms_per_example"),
        ("p95 byte", "toklen_mmlu_p95"),  # heuristic: MMLU rows; KMMLU rows fall back below
        ("frac>7B", "toklen_mmlu_frac_gt_hn"),
    ]

    header = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]

    for r in rows:
        # If the row is a KMMLU one, swap the toklen lookup to kmmlu_*.
        if r.get("task") == "kmmlu":
            col_overrides = {"toklen_mmlu_p95": "toklen_kmmlu_p95",
                             "toklen_mmlu_frac_gt_hn": "toklen_kmmlu_frac_gt_hn"}
        else:
            col_overrides = {}
        cells = []
        for _, key in cols:
            real_key = col_overrides.get(key, key)
            val = r.get(real_key, "")
            if isinstance(val, float):
                val = f"{val:.4f}" if "accuracy" in real_key or "frac" in real_key else f"{val:.2f}"
            cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "logs" / "suite_manifest.tsv",
        help="Path to the suite manifest TSV (default: $PROJECT_ROOT/logs/suite_manifest.tsv).",
    )
    parser.add_argument(
        "--logs",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "logs",
        help="Directory containing slurm .out files (default: $PROJECT_ROOT/logs).",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "suite_results.csv",
    )
    parser.add_argument(
        "--out_md",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "suite_results.md",
    )
    args = parser.parse_args()

    rows = aggregate(args.manifest, args.logs)
    write_csv(rows, args.out_csv)
    write_markdown(rows, args.out_md)
    print(f"Wrote {len(rows)} rows → {args.out_csv}")
    print(f"Wrote Markdown summary → {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
