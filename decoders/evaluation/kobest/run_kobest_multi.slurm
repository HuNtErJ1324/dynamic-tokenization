#!/bin/bash
#SBATCH --job-name=kobest-multi
#SBATCH --output=logs/kobest-%x-%j.out
#SBATCH --error=logs/kobest-%x-%j.err
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
TASK="${TASK:-copa}"
EXP_TYPE="${EXP_TYPE:-plain}"
THRESHOLD="${THRESHOLD:-3.0}"
OUTPUT="${OUTPUT:-/gpfs/projects/stf/kahaby/dynamic-tokenization-main/results/kobest_multi_results.json}"

echo "Model=$MODEL  task=$TASK  exp_type=$EXP_TYPE"

START=$(date +%s)
srun python multi_model_eval.py \
    --model "$MODEL" \
    --exp_type "$EXP_TYPE" \
    --task "$TASK" \
    --max_examples 0 \
    --threshold "$THRESHOLD" \
    --korean_prompt \
    --score_by_text \
    --output "$OUTPUT"

END=$(date +%s)
printf '[done] elapsed=%ss model=%s task=%s exp=%s\n' "$((END-START))" "$MODEL" "$TASK" "$EXP_TYPE"
