#!/usr/bin/env bash
#
# run_sweep.sh — submit a declarative hyperparameter sweep for dynamic tokenization.
#
# Generalizes scripts/run_suite.sh: instead of a hardcoded 16-cell matrix, read a
# TSV sweep spec (sweeps/<name>.tsv) and submit one slurm job per row, resolving
# operating-point references against data/operating_points/ops_*.json (produced by
# decoders/evaluation/compute_operating_points.py). Prints every sbatch command
# first; only submits with --submit. Job ids are appended to a manifest so
# scripts/aggregate_results.py can join slurm output back to (sweep, method).
#
# TSV columns (tab-separated, header row required):
#   tag               short label for this cell (-> job name + manifest)
#   task              mmlu | kmmlu
#   exp_type          plain | original_tk_hypernet | lp_tk_hypernet | dynamic_bpe |
#                     entropy_split | dynamic_bpe_entropy_split
#                     (NOTE: fvt / fvt_dynamic_bpe are not yet wired into the decoder
#                      evaluator — see the deferred S6 task.)
#   merges_source     integer (e.g. 100) | op:<pct>[_<boundary>] | 0
#                       op:30          -> M30 for THIS row's boundary
#                       op:30_superbpe -> M30 for the superbpe curve explicitly
#   boundary          pretokens | word | word_hyphen | sentence | superbpe
#   transition_point  integer | frac:<f> (fraction of resolved merges, e.g. frac:0.5) | 0
#   threshold         entropy_threshold for *_entropy_split exp_types (float)
#   batch_size        integer
#   max_examples      integer (0 = full test set; sweeps usually 1500)
#   split_strategy    entropy | random   (random = mechanism control for S8; default entropy)
#
# An empty cell or "-" falls back to a default
# (pretokens / 0 / 3.0 / 4 / 1500 / entropy).
#
# Usage:
#   # 0) one-time operating points (CPU is fine):
#   python decoders/evaluation/compute_operating_points.py
#   # 1) preview (no sbatch calls):
#   bash scripts/run_sweep.sh sweeps/S2_threshold.tsv
#   # 2) submit for real:
#   bash scripts/run_sweep.sh sweeps/S2_threshold.tsv --submit
#
# Optional env vars:
#   OPS_DIR    default $PROJECT_ROOT/data/operating_points
#   MANIFEST   default $PROJECT_ROOT/logs/sweep_manifest.tsv

set -euo pipefail

# --- locate repo root --------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

OPS_DIR="${OPS_DIR:-$PROJECT_ROOT/data/operating_points}"
MANIFEST="${MANIFEST:-$PROJECT_ROOT/logs/sweep_manifest.tsv}"
mkdir -p "$(dirname "$MANIFEST")"

# --- parse args (spec path + optional --submit, any order) -------------------
SUBMIT=0
SPEC=""
for arg in "$@"; do
    if [[ "$arg" == "--submit" ]]; then
        SUBMIT=1
    else
        SPEC="$arg"
    fi
done
if [[ -z "$SPEC" || ! -f "$SPEC" ]]; then
    echo "ERROR: pass a sweep spec TSV as an argument (e.g. sweeps/S2_threshold.tsv)" >&2
    exit 1
fi
SWEEP_NAME="$(basename "$SPEC" .tsv)"

# --- helpers -----------------------------------------------------------------
# echo $2 if $1 is empty or "-", else $1
_default() { if [[ -z "${1:-}" || "$1" == "-" ]]; then echo "$2"; else echo "$1"; fi; }

# Resolve op:<pct>[_<boundary>] (or a literal int) to a merge count.
resolve_merges() {
    local src=$1 task=$2 lang=$3 row_boundary=$4
    if [[ "$src" == op:* ]]; then
        local spec=${src#op:}              # e.g. 30  or  30_superbpe
        local pct=${spec%%_*}              # 30
        local bnd
        if [[ "$spec" == *_* ]]; then bnd=${spec#*_}; else bnd=$row_boundary; fi
        local f="$OPS_DIR/ops_${task}_${lang}_${bnd}.json"
        if [[ ! -f "$f" ]]; then
            echo "ERROR: missing operating-point file $f" >&2
            echo "       Run: python decoders/evaluation/compute_operating_points.py --boundary $bnd" >&2
            exit 1
        fi
        python3 -c "import json; d=json.load(open(r'$f')); print(int(d['operating_points']['$pct']['merges']))"
    else
        echo "$src"
    fi
}

# --- manifest header on first submission -------------------------------------
if [[ $SUBMIT -eq 1 && ! -s "$MANIFEST" ]]; then
    printf 'submitted_at\tjob_id\tsweep\tmethod\ttask\texp_type\tmerges\tboundary\ttransition_point\tentropy_threshold\tbatch_size\tmax_examples\tsplit_strategy\top\n' \
        > "$MANIFEST"
fi

# --- iterate spec rows -------------------------------------------------------
n_cells=0
# `|| [[ -n "$tag" ]]` so the last line is processed even without a trailing newline.
while IFS=$'\t' read -r tag task exp_type merges_source boundary transition_point threshold batch_size max_examples split_strategy || [[ -n "${tag:-}" ]]; do
    [[ -z "${tag:-}" ]] && continue
    [[ "$tag" == \#* ]] && continue       # comment line
    [[ "$tag" == "tag" ]] && continue     # header

    boundary=$(_default "${boundary:-}" pretokens)
    transition_point=$(_default "${transition_point:-}" 0)
    threshold=$(_default "${threshold:-}" 3.0)
    batch_size=$(_default "${batch_size:-}" 4)
    max_examples=$(_default "${max_examples:-}" 1500)
    split_strategy=$(_default "${split_strategy:-}" entropy)
    merges_source=$(_default "${merges_source:-}" 0)

    # task -> slurm script + lang
    slurm_script=""; lang=""
    if [[ "$task" == "mmlu" ]]; then
        slurm_script="decoders/evaluation/mmlu/run_mmlu.slurm"; lang="en"
    elif [[ "$task" == "kmmlu" ]]; then
        slurm_script="decoders/evaluation/kmmlu/run_kmmlu.slurm"; lang="ko"
    else
        echo "ERROR: unknown task '$task' in row tag=$tag" >&2; exit 1
    fi

    merges=$(resolve_merges "$merges_source" "$task" "$lang" "$boundary")

    # transition_point frac:<f> -> round(f * merges)
    if [[ "$transition_point" == frac:* ]]; then
        frac=${transition_point#frac:}
        transition_point=$(python3 -c "print(int(round($frac * $merges)))")
    fi

    envs=(
        "PROJECT_ROOT=$PROJECT_ROOT"
        "EXP_TYPE=$exp_type"
        "MERGES=$merges"
        "BOUNDARY=$boundary"
        "TRANSITION_POINT=$transition_point"
        "ENTROPY_THRESHOLD=$threshold"
        "BATCH_SIZE=$batch_size"
        "MAX_EXAMPLES=$max_examples"
        "SPLIT_STRATEGY=$split_strategy"
    )
    if [[ "$task" == "kmmlu" ]]; then envs+=("LNG=$lang"); fi

    job_name="${task}_${SWEEP_NAME}_${tag}"
    env_str=$(IFS=,; echo "${envs[*]}")
    cmd="sbatch --job-name=$job_name --export=ALL,$env_str $slurm_script"

    printf '[%s/%s/%s] exp=%s merges=%s boundary=%s tp=%s thr=%s bs=%s n=%s split=%s\n' \
        "$SWEEP_NAME" "$tag" "$task" "$exp_type" "$merges" "$boundary" \
        "$transition_point" "$threshold" "$batch_size" "$max_examples" "$split_strategy"
    echo "    $cmd"
    n_cells=$((n_cells + 1))

    if [[ $SUBMIT -eq 1 ]]; then
        jobid=$(sbatch --job-name="$job_name" --export=ALL,"$env_str" "$slurm_script" | awk '{print $NF}')
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date -Iseconds)" "$jobid" "$SWEEP_NAME" "$tag" "$task" "$exp_type" \
            "$merges" "$boundary" "$transition_point" "$threshold" \
            "$batch_size" "$max_examples" "$split_strategy" "$merges_source" \
            >> "$MANIFEST"
    fi
done < <(tr -d '\r' < "$SPEC")   # tr strips CR so CRLF specs (e.g. authored on Windows) parse cleanly

echo ""
echo "Sweep '$SWEEP_NAME': $n_cells cell(s)."
if [[ $SUBMIT -eq 0 ]]; then
    echo "Preview only — re-run with --submit to queue. Manifest target: $MANIFEST"
else
    echo "Submitted. Manifest: $MANIFEST"
fi
