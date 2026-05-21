#!/bin/bash
# setup_tillicum.sh — Run this script on Tillicum (klone.hyak.uw.edu) to set up
# the dynamic-tokenization project from scratch.
#
# Usage:
#   1. Upload this repo to Tillicum:
#      scp -r dynamic-tokenization-main.zip <UWNetID>@klone.hyak.uw.edu:$SCRATCH/
#   2. SSH into Tillicum and run:
#      bash $SCRATCH/setup_tillicum.sh
#
set -euo pipefail

PROJECT_DIR="${SCRATCH}/dynamic-tokenization"
HF_CACHE="${SCRATCH}/hf_cache"

echo "=============================="
echo " Dynamic Tokenization Setup"
echo "=============================="
echo "Project dir : $PROJECT_DIR"
echo "HF cache    : $HF_CACHE"
echo ""

# ── 1. Unzip project ──────────────────────────────────────────────────────────
if [ ! -d "$PROJECT_DIR" ]; then
    ZIP="${SCRATCH}/dynamic-tokenization-main.zip"
    if [ ! -f "$ZIP" ]; then
        echo "[ERROR] Upload the zip first:"
        echo "  scp dynamic-tokenization-main.zip <UWNetID>@klone.hyak.uw.edu:\$SCRATCH/"
        exit 1
    fi
    echo "[1/5] Extracting zip ..."
    unzip -q "$ZIP" -d "$SCRATCH"
    mv "$SCRATCH/dynamic-tokenization-main" "$PROJECT_DIR"
    echo "      → $PROJECT_DIR"
else
    echo "[1/5] Project directory already exists, skipping unzip."
fi

# ── 2. Set up HF cache on scratch (avoids home quota issues) ──────────────────
echo "[2/5] Configuring HuggingFace cache ..."
mkdir -p "$HF_CACHE"
if ! grep -q "HF_HOME" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# HuggingFace cache on scratch (set by setup_tillicum.sh)" >> ~/.bashrc
    echo "export HF_HOME=\"$HF_CACHE\"" >> ~/.bashrc
    echo "export TRANSFORMERS_CACHE=\"$HF_CACHE\"" >> ~/.bashrc
    echo "export HF_DATASETS_CACHE=\"$HF_CACHE/datasets\"" >> ~/.bashrc
fi
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets"
echo "      → $HF_CACHE"

# ── 3. Create conda environment ───────────────────────────────────────────────
echo "[3/5] Setting up conda environment 'dyntok' ..."
module load conda

if conda info --envs | grep -q "^dyntok "; then
    echo "      → 'dyntok' already exists, skipping creation."
else
    conda create -n dyntok python=3.10 -y
    echo "      → Created 'dyntok' environment."
fi

# Activate and install dependencies
conda run -n dyntok pip install --upgrade pip
conda run -n dyntok pip install -r "$PROJECT_DIR/requirements.curated.txt"
echo "      → Dependencies installed."

# ── 4. HuggingFace login ─────────────────────────────────────────────────────
echo "[4/5] HuggingFace login ..."
echo "      You need a HF token with access to mistralai/Mistral-7B-v0.1."
echo "      Get one from: https://huggingface.co/settings/tokens"
echo "      Also request access at: https://huggingface.co/mistralai/Mistral-7B-v0.1"
echo ""
echo "      Run the following to log in (or skip if already logged in):"
echo "        conda activate dyntok && huggingface-cli login"
echo ""

# ── 5. Create log directories ─────────────────────────────────────────────────
echo "[5/5] Creating log directories ..."
mkdir -p "$PROJECT_DIR/logs"
echo "      → $PROJECT_DIR/logs"

echo ""
echo "=============================="
echo " Setup complete!"
echo "=============================="
echo ""
echo "Next steps:"
echo ""
echo "  conda activate dyntok"
echo "  huggingface-cli login          # if not already done"
echo ""
echo "  cd $PROJECT_DIR"
echo ""
echo "  # Run KMMLU (plain baseline):"
echo "  EXP_TYPE=plain BATCH_SIZE=4 sbatch decoders/evaluation/kmmlu/run_kmmlu.slurm"
echo ""
echo "  # Run KMMLU (SuperBPE):"
echo "  EXP_TYPE=dynamic_bpe MERGES=1000 BOUNDARY=superbpe BATCH_SIZE=4 \\"
echo "    sbatch decoders/evaluation/kmmlu/run_kmmlu.slurm"
echo ""
echo "  # Run KoBEST (plain, all tasks):"
echo "  EXP_TYPE=plain BATCH_SIZE=4 sbatch decoders/evaluation/kobest/run_kobest.slurm"
echo ""
echo "  # Run HRM8K (plain, English + Korean):"
echo "  EXP_TYPE=plain BATCH_SIZE=4 LANG=both sbatch decoders/evaluation/hrm8k/run_hrm8k.slurm"
echo ""
echo "  # Monitor jobs:"
echo "  squeue -u \$USER"
echo "  tail -f $PROJECT_DIR/logs/*.out"
