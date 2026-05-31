#!/bin/bash
#SBATCH --job-name=kmmlu-multi
#SBATCH --output=logs/kmmlu-%x-%j.out
#SBATCH --error=logs/kmmlu-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --qos=normal
#SBATCH --nodelist=g023

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/stf/kahaby/dynamic-tokenization-main}"
cd "$PROJECT_ROOT"

export PATH="/gpfs/scrubbed/kahaby/conda/dyntok/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/gpfs/scrubbed/kahaby/hf_cache}"

nvidia-smi || true
python -c "import torch; print('cuda:', torch.cuda.is_available())"

cd "$PROJECT_ROOT/decoders/evaluation/kobest"
mkdir -p logs

MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
EXP_TYPE="${EXP_TYPE:-plain}"
THRESHOLD="${THRESHOLD:-3.0}"
SCORE_BY_TEXT="${SCORE_BY_TEXT:-1}"
KOREAN_PROMPT="${KOREAN_PROMPT:-1}"
OUTPUT="${OUTPUT:-/gpfs/projects/stf/kahaby/dynamic-tokenization-main/results/kmmlu_multi_results.json}"

EXTRA_ARGS=""
[ "$SCORE_BY_TEXT" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --score_by_text"
[ "$KOREAN_PROMPT" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --korean_prompt"

echo "Model=$MODEL  exp_type=$EXP_TYPE  threshold=$THRESHOLD"

START=$(date +%s)
srun python multi_model_eval.py \
    --model "$MODEL" \
    --exp_type "$EXP_TYPE" \
    --task kmmlu \
    --max_examples 4500 \
    --threshold "$THRESHOLD" \
    --output "$OUTPUT" \
    $EXTRA_ARGS \
    --no_wandb 2>/dev/null || true

END=$(date +%s)
printf '[done] elapsed=%ss model=%s exp=%s\n' "$((END-START))" "$MODEL" "$EXP_TYPE"
