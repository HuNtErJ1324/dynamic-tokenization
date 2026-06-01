# Dynamic-tokenization experiment sweeps

Declarative grids for the rigorous sweep matrix. Each `S*.tsv` is consumed by
[`scripts/run_sweep.sh`](../scripts/run_sweep.sh); one row = one slurm job.
`generate_specs.py` is the single source of truth — edit the grids there and
re-run it to regenerate the TSVs.

**Scope:** headline = entropy-split; benchmarks = MMLU (en) + KMMLU (ko);
inference-time only (frozen Mistral-7B + ZeTT hypernetwork). Sweep cells run on a
seeded 1,500-example subsample (`max_examples=1500`); winning configs are re-run
on the full test sets for the final table.

## Workflow

```bash
# 0) one-time: operating points M10..M50 (CPU, no GPU). Default = pretokens + superbpe.
python decoders/evaluation/compute_operating_points.py
#    S4 also needs the other three boundaries:
python decoders/evaluation/compute_operating_points.py --boundary word
python decoders/evaluation/compute_operating_points.py --boundary word_hyphen
python decoders/evaluation/compute_operating_points.py --boundary sentence

# 1) (re)generate the TSV specs. Reads the ops JSONs from step 0 and tailors each
#    boundary's operating points to what's achievable (drops clamped duplicates), so
#    run this AFTER compute_operating_points. Without the JSONs it falls back to the
#    full ladder with a warning.
python sweeps/generate_specs.py

# 2) preview a sweep (prints sbatch commands, submits nothing)
bash scripts/run_sweep.sh sweeps/S2_threshold.tsv

# 3) submit it. Each cell requests WALLTIME (default 1h). The cluster estimates cost
#    from REQUESTED time, so keep it short for subsample sweeps (the 12h slurm default
#    over-reserves and trips the account budget). Raise it for full-set runs:
#    WALLTIME=12:00:00 bash scripts/run_sweep.sh ... --submit
bash scripts/run_sweep.sh sweeps/S2_threshold.tsv --submit

# 4) after jobs finish: aggregate + plot
python scripts/aggregate_results.py --manifest logs/sweep_manifest.tsv
python scripts/plot_pareto.py
```

## The sweeps

| Spec | Purpose | Key axis |
|---|---|---|
| `S1_pareto.tsv` | accuracy vs compression Pareto | merges (M10–M50) × {M, U, SM, SMsup} + baselines |
| `S2_threshold.tsv` | **headline** entropy-threshold sweep | threshold ∈ {1,2,3,4,5,6,8,no-split} × {entropy_split, SM, SMsup} |
| `S3_threshold_x_merges.tsv` | does optimal threshold shift with compression? | merges {M20,M30,M40} × threshold {2,3,4,6} (SM) |
| `S4_boundary.tsv` | boundary-constraint ablation @ M30 | {pretokens, word, word_hyphen, sentence, superbpe} |
| `S5_transition_point.tsv` | SuperBPE warm-up schedule @ M30(superbpe) | transition_point ∈ {0, M/4, M/2, 3M/4} |
| `S7_batch_size.tsv` | per-batch-merging sensitivity @ M30 | batch_size ∈ {1,4,8,16,32} |
| `S8_split_strategy.tsv` | mechanism control | entropy-split vs random-split at matched split_rate |

`S6` (FVT vs hypernetwork) is **deferred**: `fvt`/`fvt_dynamic_bpe` are
implemented in `tokenizations/tokenization_utils.py` but not yet routed in the
decoder evaluator (`evaluate_model` raises `NotImplementedError`).

## TSV column reference

`tag, task, exp_type, merges_source, boundary, transition_point, threshold, batch_size, max_examples, split_strategy`

- `merges_source`: integer, or `op:<pct>` (M<pct> for the row's boundary), or `op:<pct>_<boundary>`.
- `transition_point`: integer, or `frac:<f>` (fraction of resolved merges).
- `split_strategy`: `entropy` (default) or `random` (S8 control).
- empty / `-` → run_sweep.sh default (`pretokens / 0 / 3.0 / 4 / 1500 / entropy`).
