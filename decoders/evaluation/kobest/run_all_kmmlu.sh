#!/bin/bash
# Run KMMLU (all subjects) for all models — plain + entropy_split
set -e
cd "$(dirname "$0")"
OUT="../../../results/kmmlu_results.json"

MODELS=(
  "skt/ko-gpt-trinity-1.2B-v0.5"
  "EleutherAI/polyglot-ko-1.3b"
  "EleutherAI/polyglot-ko-5.8b"
  "kfkas/Llama-2-ko-7b-Chat"
  "mistralai/Mistral-7B-v0.1"
  "beomi/Llama-3-KoEn-8B"
  "Qwen/Qwen2.5-7B"
)

for MODEL in "${MODELS[@]}"; do
  SHORT=$(echo "$MODEL" | cut -d'/' -f2)
  echo ""
  echo "=============================="
  echo "MODEL: $SHORT"
  echo "=============================="

  echo "[plain]"
  python multi_model_eval.py \
    --model "$MODEL" --exp_type plain \
    --task kmmlu --kmmlu_subject all \
    --max_examples 0 \
    --korean_prompt --score_by_text \
    --output "$OUT" 2>&1 | grep -E "Result|Error|\[" | tail -5

  echo "[entropy_split thr=3.0]"
  python multi_model_eval.py \
    --model "$MODEL" --exp_type entropy_split \
    --task kmmlu --kmmlu_subject all \
    --max_examples 0 --threshold 3.0 \
    --korean_prompt --score_by_text \
    --output "$OUT" 2>&1 | grep -E "Result|Error|\[" | tail -5
done

echo ""
echo "All done."
