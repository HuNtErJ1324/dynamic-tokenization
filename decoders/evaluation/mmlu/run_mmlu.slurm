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
# Conda is already in the inherited login env on Tillicum; just enable `conda activate`
# in this non-interactive shell and switch to the project env.
module load dyntok
conda activate dyntok

# Avoid HF tokenizers parallelism warning under DataLoader
export TOKENIZERS_PARALLELISM=false

# Optional: cache HF datasets/models on fast local scratch instead of $HOME
# export HF_HOME="$SCRATCH/hf_cache"
# export TRANSFORMERS_CACHE="$HF_HOME"
# export HF_DATASETS_CACHE="$HF_HOME/datasets"

# Optional: wandb login (or `wandb login` once on a login node)
# export WANDB_API_KEY=...
# export WANDB_MODE=offline   # if compute nodes have no internet

nvidia-smi || true
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'device count:', torch.cuda.device_count())"

# --- run ---------------------------------------------------------------------
# The script does `from evaluation_utils import ...`, so it must run from its own dir.
cd "$PROJECT_ROOT/decoders/evaluation/mmlu"

# Choose the experiment via env var: plain | original_tk_hypernet | dynamic_bpe
EXP_TYPE="${EXP_TYPE:-plain}"
MERGES="${MERGES:-1000}"     # only used by dynamic_bpe
BATCH_SIZE="${BATCH_SIZE:-4}"
# Merge boundary for dynamic_bpe: pretokens | word | word_hyphen | sentence | superbpe
BOUNDARY="${BOUNDARY:-pretokens}"
# SuperBPE two-stage schedule: first N merges use strict 'pretokens' boundary,
# remainder use $BOUNDARY. 0 disables the warm-up phase.
TRANSITION_POINT="${TRANSITION_POINT:-0}"

echo "Running MMLU eval: exp_type=$EXP_TYPE, merges=$MERGES, batch_size=$BATCH_SIZE, boundary=$BOUNDARY, transition_point=$TRANSITION_POINT"

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
    --five_shot \
    --no_wandb

END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
printf '[slurm-latency] end_time=%s (epoch=%s)\n' "$(date -Iseconds)" "$END_EPOCH"
printf '[slurm-latency] total_elapsed=%ss (%dh %dm %ds) exp_type=%s merges=%s batch_size=%s\n' \
    "$ELAPSED" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) \
    "$EXP_TYPE" "$MERGES" "$BATCH_SIZE"
