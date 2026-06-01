#!/bin/bash
#SBATCH --job-name=mmlu-mistral
#SBATCH --output=logs/mmlu-%x-%j.out
#SBATCH --error=logs/mmlu-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --qos=normal
# #SBATCH --account=<allocation>   # uncomment if your Tillicum project requires one
#
# Tillicum (Hyak) scheduling model — see https://hyak.uw.edu/docs/tillicum/scheduling-jobs
#   - QoS-based, no partitions. Default --qos=normal: 24h max walltime, up to 16 GPUs/job.
#   - A GPU request is mandatory (jobs without GPUs are rejected). Nodes have 8x H200.
#   - Each GPU grants 8 CPUs + ~200G system RAM; to get more CPU/RAM, request more GPUs.
#   - No preemptible checkpoint partition, so --requeue / --signal handling is not needed.

set -euo pipefail

# --- paths -------------------------------------------------------------------
# Root of the repo on the HPC (the directory containing tokenizations/, encoders/, decoders/).
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/stf/dyntok/dynamic-tokenization}"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

# --- env ---------------------------------------------------------------------
# Only `module load conda` if it isn't already loaded. sbatch defaults to
# --export=ALL, so if the submitter had conda loaded interactively, CONDA_PREFIX
# will already be set here and a fresh `module load conda` would error with
# "Conda is already active".
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    module load conda
fi
conda activate dyntok

# Avoid HF tokenizers parallelism warning under DataLoader
export TOKENIZERS_PARALLELISM=false

# HuggingFace auth + offline cache. KMMLU jobs enumerate ~45 subject configs, which
# rate-limits the HF API (HTTP 429) without a token. Pre-cache once with
# scripts/precache_data.py, then submit with HF_HUB_OFFLINE=1 so jobs read the local
# cache and never call the API. These inherit from the submitter via --export=ALL.
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"

# Optional: wandb login (or `wandb login` once on a login node)
# export WANDB_API_KEY=...
# export WANDB_MODE=offline   # if compute nodes have no internet

nvidia-smi || true
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'device count:', torch.cuda.device_count())"

# --- run ---------------------------------------------------------------------
# The script does `from evaluation_utils import ...`, so it must run from its own dir.
cd "$PROJECT_ROOT/decoders/evaluation/mmlu"

# Choose the experiment via env var:
#   plain | original_tk_hypernet | dynamic_bpe | entropy_split | dynamic_bpe_entropy_split
EXP_TYPE="${EXP_TYPE:-plain}"
MERGES="${MERGES:-1000}"     # only used by dynamic_bpe* exp_types
BATCH_SIZE="${BATCH_SIZE:-4}"
# Merge boundary for dynamic_bpe: pretokens | word | word_hyphen | sentence | superbpe
BOUNDARY="${BOUNDARY:-pretokens}"
# SuperBPE two-stage schedule: first N merges use strict 'pretokens' boundary,
# remainder use $BOUNDARY. 0 disables the warm-up phase.
TRANSITION_POINT="${TRANSITION_POINT:-0}"
# Entropy threshold for entropy_split / dynamic_bpe_entropy_split.
ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:-3.0}"
# If >0, evaluate on a seeded random subsample of this many test examples (cheap sweeps).
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
# Token-split selection for *_entropy_split exp_types: entropy (method) | random (S8 control).
SPLIT_STRATEGY="${SPLIT_STRATEGY:-entropy}"

echo "Running MMLU eval: exp_type=$EXP_TYPE, merges=$MERGES, batch_size=$BATCH_SIZE, boundary=$BOUNDARY, transition_point=$TRANSITION_POINT, entropy_threshold=$ENTROPY_THRESHOLD, max_examples=$MAX_EXAMPLES, split_strategy=$SPLIT_STRATEGY"

# --- latency tracking --------------------------------------------------------
START_EPOCH=$(date +%s)
echo "[slurm-latency] start_time=$(date -Iseconds) (epoch=$START_EPOCH)"

srun python mistral_mmlu.py \
    --exp_type "$EXP_TYPE" \
    --eval_type original \
    --ds_subject all \
    --batch_size "$BATCH_SIZE" \
    --merges "$MERGES" \
    --bpe_tokenizer_boundary "$BOUNDARY" \
    --transition_point "$TRANSITION_POINT" \
    --entropy_threshold "$ENTROPY_THRESHOLD" \
    --max_examples "$MAX_EXAMPLES" \
    --split_strategy "$SPLIT_STRATEGY" \
    --five_shot \
    --no_wandb

END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
printf '[slurm-latency] end_time=%s (epoch=%s)\n' "$(date -Iseconds)" "$END_EPOCH"
printf '[slurm-latency] total_elapsed=%ss (%dh %dm %ds) exp_type=%s merges=%s batch_size=%s\n' \
    "$ELAPSED" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) \
    "$EXP_TYPE" "$MERGES" "$BATCH_SIZE"
