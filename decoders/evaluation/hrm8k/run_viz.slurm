#!/bin/bash
#SBATCH --job-name=viz-dyn-bpe
#SBATCH --output=logs/viz-%j.out
#SBATCH --error=logs/viz-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=0:30:00
#SBATCH --qos=normal

export PATH="/gpfs/scrubbed/kahaby/conda/dyntok/bin:$PATH"
export PYTHONHOME="/gpfs/scrubbed/kahaby/conda/dyntok"
export HF_HOME="/gpfs/scrubbed/kahaby/hf_cache"

THRESHOLD="${THRESHOLD:-3.0}"

cd /gpfs/projects/stf/kahaby/dynamic-tokenization-main/decoders/evaluation/hrm8k

python visualize_dynamic_bpe_split.py --merges 1000 --threshold "$THRESHOLD" --lang ko
