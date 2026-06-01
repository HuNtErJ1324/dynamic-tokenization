"""
Aggregate slurm logs from a run of scripts/run_sweep.sh (or the legacy
scripts/run_suite.sh) into a single results table. Reads the manifest
(job_id <-> method <-> config) and pairs every job with its matching slurm log
under logs/, parsing stdout for accuracy, latency, and merged-token-length stats.

Realized sequence-reduction % is filled in from the operating-point JSONs
(data/operating_points/ops_*.json) when a row references one via op:<pct> — this
is the simulated reduction on the operating-point set, used as the x-axis for the
Pareto plots. Rows with merges=0 (plain / original_tk_hypernet) get 0%.

Outputs:
  - results/suite_results.csv     — one row per cell (all columns)
  - results/suite_results.md      — tighter Markdown table for the report

Usage:
    python scripts/aggregate_results.py                                   # sweep manifest
    python scripts/aggregate_results.py --manifest logs/suite_manifest.tsv  # legacy 16-cell suite
"""

from __future__ import annotations

import argparse
import csv
import json
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
    """Read the TSV manifest produced by run_sweep.sh / run_suite.sh --submit."""
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {path}. "
            "Run scripts/run_sweep.sh --submit first, or pass --manifest explicitly."
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
        candidates = list(logs_dir.glob(f"*{job_id}*.out"))
    if not candidates:
        return None
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


def _seq_reduction_pct(entry: dict, ops_dir: Path) -> tuple[Optional[float], Optional[float]]:
    """Best-effort (target, realized) sequence-reduction % for a manifest row.

    For op:<pct>[_<boundary>] rows we read the simulated achieved_reduction_pct
    from the matching operating-point JSON (proxy for realized reduction). Rows
    with merges=0 are 0%. Anything else returns (None, None).
    """
    op = (entry.get("op") or "").strip()
    merges = (entry.get("merges") or "").strip()

    if op.startswith("op:"):
        spec = op[3:]                       # e.g. "30" or "30_superbpe"
        pct = spec.split("_")[0]
        boundary = spec.split("_", 1)[1] if "_" in spec else (entry.get("boundary") or "pretokens")
        task = entry.get("task", "")
        lang = "en" if task == "mmlu" else "ko"
        target = float(pct) if pct.replace(".", "", 1).isdigit() else None
        f = ops_dir / f"ops_{task}_{lang}_{boundary}.json"
        if f.exists():
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                rec = d.get("operating_points", {}).get(pct, {})
                ach = rec.get("achieved_reduction_pct")
                return target, (float(ach) if ach is not None else target)
            except Exception:
                return target, target
        return target, target

    if merges in ("0", ""):
        return 0.0, 0.0
    return None, None


def aggregate(manifest_path: Path, logs_dir: Path, ops_dir: Path) -> list[dict]:
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
            print(
                f"[warn] no log for job_id={job_id} method={entry.get('method')} "
                f"task={entry.get('task')}",
                flush=True,
            )
        tgt, red = _seq_reduction_pct(entry, ops_dir)
        if tgt is not None:
            merged["target_reduction_pct"] = tgt
        if red is not None:
            merged["seq_reduction_pct"] = red
        rows.append(merged)
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    # Stable header: manifest/config columns first, then derived + numeric metrics,
    # then any extra toklen_* keys, then log_path.
    base_cols = [
        "submitted_at", "job_id", "sweep", "method", "task", "exp_type", "merges",
        "boundary", "transition_point", "entropy_threshold", "batch_size",
        "max_examples", "split_strategy", "op",
        "target_reduction_pct", "seq_reduction_pct",
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


def _fmt(key: str, val) -> str:
    if isinstance(val, float):
        if "accuracy" in key or key == "split_rate" or "frac" in key:
            return f"{val:.4f}"
        if "reduction_pct" in key:
            return f"{val:.1f}"
        return f"{val:.2f}"
    return str(val)


def write_markdown(rows: list[dict], out_path: Path) -> None:
    """Tighter, task-agnostic Markdown table for the report."""
    cols = [
        ("Sweep", "sweep"),
        ("Method", "method"),
        ("Task", "task"),
        ("exp_type", "exp_type"),
        ("Boundary", "boundary"),
        ("Merges", "merges"),
        ("Thr", "entropy_threshold"),
        ("Split", "split_strategy"),
        ("Reduc%", "seq_reduction_pct"),
        ("Accuracy", "accuracy"),
        ("ms/ex", "ms_per_example"),
        ("split_rate", "split_rate"),
    ]
    header = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for r in rows:
        cells = [_fmt(key, r.get(key, "")) for _, key in cols]
        lines.append("| " + " | ".join(cells) + " |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "logs" / "sweep_manifest.tsv",
        help="Path to the manifest TSV (default: logs/sweep_manifest.tsv; "
        "pass logs/suite_manifest.tsv for the legacy 16-cell suite).",
    )
    parser.add_argument(
        "--logs",
        type=Path,
        default=repo_root / "logs",
        help="Directory containing slurm .out files (default: logs/).",
    )
    parser.add_argument(
        "--ops_dir",
        type=Path,
        default=repo_root / "data" / "operating_points",
        help="Directory with ops_*.json for realized sequence-reduction lookup.",
    )
    parser.add_argument("--out_csv", type=Path, default=repo_root / "results" / "suite_results.csv")
    parser.add_argument("--out_md", type=Path, default=repo_root / "results" / "suite_results.md")
    args = parser.parse_args()

    rows = aggregate(args.manifest, args.logs, args.ops_dir)
    write_csv(rows, args.out_csv)
    write_markdown(rows, args.out_md)
    print(f"Wrote {len(rows)} rows -> {args.out_csv}")
    print(f"Wrote Markdown summary -> {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
