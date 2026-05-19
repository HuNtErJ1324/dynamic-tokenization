#!/bin/bash

# MMLU Splitting Threshold Divergence Analysis

BATCH_SIZE="${BATCH_SIZE:-4}"
SUBJECT="$1" # If empty, we now exit

# Exit early if no subject provided
if [ -z "$SUBJECT" ]; then
    echo "No subject specified. Exiting without running divergence analysis."
    echo "Usage: SUBJECT=<subject_name> ./script.sh"
    exit 0
fi

LOG_DIR=".eval_divergence_logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/mmlu_divergence_${SUBJECT}_${TIMESTAMP}.log"

echo "START DIVERGENCE ANALYSIS"
echo "Subject: $SUBJECT"
echo "Logging to: $LOG_FILE"

python mistral_mmlu.py \
    --eval_type original \
    --batch_size "$BATCH_SIZE" \
    --split_divergence_analysis \
    --no_wandb \
    --divergence_subject "$SUBJECT" \
    2>&1 | tee "$LOG_FILE"

# Check result
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ">>> Divergence analysis completed successfully."
    echo "Results logged to: $LOG_FILE"
else
    echo "!!! Warning: Divergence analysis failed. Check logs."
fi

echo "--------------------------------------------------"