#!/bin/bash
#SBATCH --job-name=kobest-mistral
#SBATCH --output=logs/kobest-%x-%j.out
#SBATCH --error=logs/kobest-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
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

cd "$PROJECT_ROOT/decoders/evaluation/kobest"

EXP_TYPE="${EXP_TYPE:-plain}"
MERGES="${MERGES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LNG="${LNG:-ko}"
TASK="${TASK:-all}"
BOUNDARY="${BOUNDARY:-pretokens}"
TRANSITION_POINT="${TRANSITION_POINT:-0}"
ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:-3.0}"

echo "Running KoBEST eval: task=$TASK, exp_type=$EXP_TYPE, merges=$MERGES, batch_size=$BATCH_SIZE, boundary=$BOUNDARY, entropy_threshold=$ENTROPY_THRESHOLD"

START_EPOCH=$(date +%s)
echo "[slurm-latency] start_time=$(date -Iseconds) (epoch=$START_EPOCH)"

srun python mistral_kobest.py \
    --task "$TASK" \
    --exp_type "$EXP_TYPE" \
    --eval_type original \
    --batch_size "$BATCH_SIZE" \
    --merges "$MERGES" \
    --lng "$LNG" \
    --bpe_tokenizer_boundary "$BOUNDARY" \
    --transition_point "$TRANSITION_POINT" \
    --entropy_threshold "$ENTROPY_THRESHOLD" \
    --five_shot \
    --no_wandb

END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
printf '[slurm-latency] end_time=%s (epoch=%s)\n' "$(date -Iseconds)" "$END_EPOCH"
printf '[slurm-latency] total_elapsed=%ss (%dh %dm %ds) exp_type=%s task=%s boundary=%s\n' \
    "$ELAPSED" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) \
    "$EXP_TYPE" "$TASK" "$BOUNDARY"
