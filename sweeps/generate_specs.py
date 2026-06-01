"""
Generate declarative sweep specs (sweeps/S*.tsv) consumed by scripts/run_sweep.sh.

Each row is one slurm cell. This script is the single source of truth for the
experiment matrix; the emitted TSVs are committed so the exact grid we ran is
reproducible. Edit the grids here, re-run, and commit the regenerated TSVs.

Columns (see scripts/run_sweep.sh for resolution rules):
  tag, task, exp_type, merges_source, boundary, transition_point, threshold,
  batch_size, max_examples, split_strategy

merges_source 'op:<pct>' resolves against the row's boundary operating-point
JSON at submit time; transition_point 'frac:<f>' is a fraction of resolved
merges; '-' means "use the run_sweep.sh default".

Scope (per user): headline = entropy-split; benchmarks = MMLU(en)+KMMLU(ko);
inference-time only. FVT (S6) is deferred (not yet wired into the decoder
evaluator), so it is intentionally absent here.

Run:  python sweeps/generate_specs.py
"""

from pathlib import Path

HEADER = [
    "tag", "task", "exp_type", "merges_source", "boundary",
    "transition_point", "threshold", "batch_size", "max_examples", "split_strategy",
]

TASKS = ["mmlu", "kmmlu"]
N = 1500           # sweep subsample size (full set reserved for final confirmation)
OPS = [10, 20, 30, 40, 50]
THR = "3.0"        # default entropy threshold for SM in the Pareto sweep

OUT = Path(__file__).resolve().parent


def row(tag, task, exp, merges, boundary="pretokens", tp=0, thr="-", bs=4, n=N, split="-"):
    return [tag, task, exp, merges, boundary, tp, thr, bs, n, split]


def write(name, rows):
    lines = ["\t".join(HEADER)] + ["\t".join(str(x) for x in r) for r in rows]
    # newline="\n": keep specs LF-only so bash `read` doesn't capture a trailing \r
    # in the last column when this runs on Windows.
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {name}: {len(rows)} cells")


# ── S1: primary Pareto — accuracy vs compression ─────────────────────────────
def s1():
    rows = []
    for task in TASKS:
        rows.append(row("B1", task, "plain", "0"))                       # baseline
        rows.append(row("B2", task, "original_tk_hypernet", "0"))        # HN, no merge
        for p in OPS:
            rows.append(row(f"M_M{p}",     task, "dynamic_bpe",               f"op:{p}", "pretokens"))
            rows.append(row(f"U_M{p}",     task, "dynamic_bpe",               f"op:{p}", "superbpe"))
            rows.append(row(f"SM_M{p}",    task, "dynamic_bpe_entropy_split", f"op:{p}", "pretokens", thr=THR))
            rows.append(row(f"SMsup_M{p}", task, "dynamic_bpe_entropy_split", f"op:{p}", "superbpe",  thr=THR))
    write("S1_pareto.tsv", rows)


# ── S2: HEADLINE — entropy-threshold sweep ───────────────────────────────────
# '100' = no-split control (threshold above max token entropy -> split_rate 0).
THRESHOLDS = ["1", "2", "3", "4", "5", "6", "8", "100"]


def s2():
    rows = []
    for task in TASKS:
        for t in THRESHOLDS:
            rows.append(row(f"S_t{t}",     task, "entropy_split",             "0",     "pretokens", thr=t))
            rows.append(row(f"SM_t{t}",    task, "dynamic_bpe_entropy_split", "op:30", "pretokens", thr=t))
            rows.append(row(f"SMsup_t{t}", task, "dynamic_bpe_entropy_split", "op:30", "superbpe",  thr=t))
    write("S2_threshold.tsv", rows)


# ── S3: 2-D interaction — threshold x merges for SM (pretokens) ───────────────
def s3():
    rows = []
    for task in TASKS:
        for p in [20, 30, 40]:
            for t in ["2", "3", "4", "6"]:
                rows.append(row(f"SM_M{p}_t{t}", task, "dynamic_bpe_entropy_split",
                                f"op:{p}", "pretokens", thr=t))
    write("S3_threshold_x_merges.tsv", rows)


# ── S4: boundary-type ablation @ M30 (per-boundary operating point) ───────────
# Needs ops_<task>_<lang>_<boundary>.json for each boundary. pretokens+superbpe
# come from the default compute_operating_points run; the other three require:
#   python decoders/evaluation/compute_operating_points.py --boundary word
#   ... --boundary word_hyphen ; ... --boundary sentence
BOUNDARIES = ["pretokens", "word", "word_hyphen", "sentence", "superbpe"]


def s4():
    rows = []
    for task in TASKS:
        for b in BOUNDARIES:
            rows.append(row(f"M30_{b}", task, "dynamic_bpe", "op:30", b))
    write("S4_boundary.tsv", rows)


# ── S5: SuperBPE warm-up (transition_point) sweep @ M30(superbpe) ─────────────
def s5():
    rows = []
    for task in TASKS:
        for f in ["0", "0.25", "0.5", "0.75"]:
            tp = "0" if f == "0" else f"frac:{f}"
            tag = f"U_tp{f.replace('.', '')}"
            rows.append(row(tag, task, "dynamic_bpe", "op:30", "superbpe", tp=tp))
    write("S5_transition_point.tsv", rows)


# ── S7: batch-size sensitivity @ M30(pretokens) ──────────────────────────────
def s7():
    rows = []
    for task in TASKS:
        for bs in [1, 4, 8, 16, 32]:
            rows.append(row(f"M30_bs{bs}", task, "dynamic_bpe", "op:30", "pretokens", bs=bs))
    write("S7_batch_size.tsv", rows)


# ── S8: mechanism control — entropy-split vs random-split ─────────────────────
def s8():
    rows = []
    for task in TASKS:
        for p in [20, 30]:
            for strat in ["entropy", "random"]:
                rows.append(row(f"SM_M{p}_{strat}", task, "dynamic_bpe_entropy_split",
                                f"op:{p}", "pretokens", thr=THR, split=strat))
    write("S8_split_strategy.tsv", rows)


def main():
    s1(); s2(); s3(); s4(); s5(); s7(); s8()
    print("\nNote: S6 (FVT vs hypernetwork) is deferred — fvt is not wired into the "
          "decoder evaluator yet.")


if __name__ == "__main__":
    main()
