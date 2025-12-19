#!/usr/bin/env bash
set -e

mkdir -p output

MODE="$1"   # baseline | cbt

if [[ "$MODE" != "baseline" && "$MODE" != "cbt" ]]; then
  echo "Usage: ./run_experiments.sh [baseline|cbt]"
  exit 1
fi

# MODEL="gemma2:9b"
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

  echo "======================================"
  echo "Running $MODE — seed $run"
  echo "======================================"

  CMD=(
    python -m cbt_llm.multiturn_convo
    # --model "$MODEL"
    --therapist_mode "$MODE"
    --turns $TURNS
    --k $K
    --seed "$seed"
    --transcript_json "output/${MODE}_transcript_${run}.json"
  )

  if [[ "$MODE" == "cbt" ]]; then
    CMD+=(--use_rag --use_schema --use_protocol)
  fi

  "${CMD[@]}"
done
