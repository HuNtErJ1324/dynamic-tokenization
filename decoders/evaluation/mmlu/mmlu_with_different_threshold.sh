#!/bin/bash

BATCH_SIZE="${BATCH_SIZE:-4}"

# Define the range of entropy thresholds to test
# These values are based on the ln(32000) vocab limit (~10.37)
THRESHOLDS=(1.0 2.0 3.0 4.0 7.0 10.0)

# Create a logs directory to store results
LOG_DIR=".eval_logs"
mkdir -p "$LOG_DIR"

echo "START EVALUATION FOR DIFFERENT THRESHOLDS"

# --- Execution Loop ---
for THRESHOLD in "${THRESHOLDS[@]}"
do
    # Create a unique log file name with threshold and timestamp
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOG_DIR/mmlu_thresh_${THRESHOLD}_${TIMESTAMP}.log"

    echo "[$(date +%T)] Testing Threshold: $THRESHOLD"
    echo "Logging to: $LOG_FILE"

    # Execution Command
    # We use 2>&1 | tee to capture errors and see progress in the terminal
    python mistral_mmlu.py \
        --eval_type original \
        --ds_subject all \
        --batch_size "$BATCH_SIZE" \
        --split \
        --split_threshold "$THRESHOLD" \
        --no_wandb 2>&1 | tee "$LOG_FILE"

    # Check if the previous command failed (e.g., OOM on your 4070 Ti)
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "!!! Warning: Run failed for threshold $THRESHOLD. Check logs."
    else
        echo ">>> Completed threshold $THRESHOLD successfully."
    fi
    
    echo "--------------------------------------------------"
done


echo "Sweep complete. All logs are located in: $LOG_DIR"