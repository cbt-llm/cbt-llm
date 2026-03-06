#!/usr/bin/env bash
set -e

# Usage: ./run_experiments.sh [baseline|cbt] [gemma|mistral|qwen|deepseek|gpt]
# Example: ./run_experiments.sh cbt mistral

MODE="$1"
MODEL_KEY="$2"

if [[ -z "$MODE" || -z "$MODEL_KEY" ]]; then
  echo "Usage: ./run_experiments.sh [baseline|cbt] [gemma|mistral|qwen|deepseek|gpt]"
  exit 1
fi

if [[ "$MODE" != "baseline" && "$MODE" != "cbt" ]]; then
  echo "Invalid mode: $MODE"
  exit 1
fi

# Model map
case "$MODEL_KEY" in
  gemma)    MODEL="gemma3:4b" ;;
  mistral)  MODEL="mistral:7b" ;;
  deepseek) MODEL="deepseek-r1:8b" ;;
  qwen) MODEL="qwen3:8b" ;;
  gpt)      MODEL="gpt-oss:20b" ;;
  *)
    echo "Unknown model key: $MODEL_KEY"
    exit 1
    ;;
esac

OUTDIR="output/${MODEL_KEY}"
mkdir -p "$OUTDIR"

TURNS=10
K=5

seeds=(
  "I’m calm most of the times but sometimes I end up blowing up."
  # "I keep overthinking everything at work, and I'm scared I'm going to get fired even though nobody said anything."
  # "I feel lonely even when I'm around people, and then I get annoyed at myself for feeling that way."
  # "I don’t want to be a burden on other people. So I never share anything and pretend I’m fine and smile."
)

for i in "${!seeds[@]}"; do
  run=$((i+1))
  seed="${seeds[$i]}"

  echo "Seed $run"

  CMD=(
    python -m cbt_llm.multiturn_convo
    --therapist_model "$MODEL"
    --therapist_mode "$MODE"
    --turns "$TURNS"
    --k "$K"
    --seed "$seed"
    --transcript_json "${OUTDIR}/${MODE}_transcript_${run}.json"
  )

  if [[ "$MODE" == "cbt" ]]; then
    CMD+=(--use_rag --use_schema --use_protocol)
  fi

  "${CMD[@]}"
done


echo "Outputs saved to ./output/${MODEL_KEY}"