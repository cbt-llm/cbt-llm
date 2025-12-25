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
  gemma)    MODEL="gemma2:9b" ;;
  mistral)  MODEL="mistral:7b-instruct" ;;
  qwen)     MODEL="qwen3:4b" ;;
  deepseek) MODEL="deepseek-r1:8b" ;;
  gpt)      MODEL="gpt-4o" ;;
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
  "I keep overthinking everything at work, and I'm scared I'm going to get fired even though nobody said anything."
  "I feel lonely even when I'm around people, and then I get annoyed at myself for feeling that way."
  "I get a tight chest before meetings and I avoid speaking, then I regret it all day."
  "I messed up one small thing yesterday and now I can't stop replaying it like it proves I'm not good enough."
  "My sleep is messed up because my brain won't shut off at night, and I wake up tired and irritated."
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