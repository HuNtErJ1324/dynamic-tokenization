#!/bin/bash
#SBATCH --job-name=hrm8k-mistral
#SBATCH --output=logs/hrm8k-%x-%j.out
#SBATCH --error=logs/hrm8k-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --qos=normal
# #SBATCH --account=<allocation>

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/stf/kahaby/dynamic-tokenization-main}"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

export PATH="/gpfs/scrubbed/kahaby/conda/dyntok/bin:$PATH"

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/gpfs/scrubbed/kahaby/hf_cache}"

nvidia-smi || true
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'device count:', torch.cuda.device_count())"

cd "$PROJECT_ROOT/decoders/evaluation/hrm8k"

EXP_TYPE="${EXP_TYPE:-plain}"
BATCH_SIZE="${BATCH_SIZE:-4}"
HRM8K_LANG="${HRM8K_LANG:-both}"  # en | ko | both
NUM_SHOTS="${NUM_SHOTS:-3}"   # 0 for zero-shot
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MERGES="${MERGES:-1000}"
LNG="${LNG:-ko}"
ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:-6.0}"

echo "Running HRM8K eval: lang=$HRM8K_LANG, exp_type=$EXP_TYPE, batch_size=$BATCH_SIZE, num_shots=$NUM_SHOTS, merges=$MERGES, lng=$LNG, entropy_threshold=$ENTROPY_THRESHOLD"

START_EPOCH=$(date +%s)
echo "[slurm-latency] start_time=$(date -Iseconds) (epoch=$START_EPOCH)"

srun python mistral_hrm8k.py \
    --lang "$HRM8K_LANG" \
    --exp_type "$EXP_TYPE" \
    --batch_size "$BATCH_SIZE" \
    --num_shots "$NUM_SHOTS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --merges "$MERGES" \
    --lng "$LNG" \
    --entropy_threshold "$ENTROPY_THRESHOLD" \
    --five_shot \
    --no_wandb

END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
printf '[slurm-latency] end_time=%s (epoch=%s)\n' "$(date -Iseconds)" "$END_EPOCH"
printf '[slurm-latency] total_elapsed=%ss (%dh %dm %ds) exp_type=%s lang=%s\n' \
    "$ELAPSED" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) \
    "$EXP_TYPE" "$HRM8K_LANG"
