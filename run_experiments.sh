#!/usr/bin/env bash
set -e

# Usage: ./run_experiments.sh [baseline|cbt|cbt_mcot] [gemma|mistral|deepseek|gpt]
# Example: ./run_experiments.sh cbt mistral

MODE="$1"
MODEL_KEY="$2"

if [[ -z "$MODE" || -z "$MODEL_KEY" ]]; then
  echo "Usage: ./run_experiments.sh [baseline|cbt|cbt_mcot] [gemma|mistral|deepseek|gpt]"
  exit 1
fi

if [[ "$MODE" != "baseline" && "$MODE" != "cbt" && "$MODE" != "cbt_mcot" ]]; then
  echo "Invalid mode: $MODE"
  exit 1
fi

# seeds_old=(
#   "I feel calm most of the time, but sometimes when small things pile up it feels like nothing is going the way it should, and I end up blowing up."
#   "I keep overthinking everything at work, and I'm scared I'm going to get fired even though nobody said anything."
#   "I feel lonely even when I'm around people, and then I get annoyed at myself for feeling that way."
#   "I don’t want to be a burden on other people. So I never share anything and pretend I’m fine and smile."
#   "Sometimes I can’t help but feel this void despite being with people who are close friends and family. I don't know if I am being ungrateful, but I don't feel like I have anyone to talk to when things get hard. I don't belong anywhere. I just don't want to wake up ever again."
# )

command -v jq >/dev/null || { echo "jq is required (brew install jq / apt-get install jq)"; exit 1; }

# Model map
case "$MODEL_KEY" in
  gemma)    MODEL="gemma3:12b" ;;
  mistral)  MODEL="mistral:7b" ;;
  deepseek) MODEL="deepseek-r1:8b" ;;
  gpt)      MODEL="gpt-oss:20b" ;;
  *)
    echo "Unknown model key: $MODEL_KEY"
    exit 1
    ;;
esac

DATA="data/processed/user_case_studies.json"
[[ -f "$DATA" ]] || { echo "Dataset not found: $DATA"; exit 1; }

OUTDIR="output/${MODEL_KEY}"
mkdir -p "$OUTDIR"

TURNS=10
K=5

# Read seeds from the combined dataset; skip rows whose seed query is null.
while IFS=$'\t' read -r id source issue seed; do
  echo "Seed $id ($source) — core issue: $issue"

  CMD=(
    python -m cbt_llm.multiturn_convo
    --therapist_model "$MODEL"
    --therapist_mode "$MODE"
    --turns "$TURNS"
    --k "$K"
    --seed "$seed"
    --core_issue "$issue"
    --transcript_json "${OUTDIR}/${MODE}_transcript_${id}.json"
  )

  if [[ "$MODE" == "cbt" || "$MODE" == "cbt_mcot" ]]; then
    CMD+=(--use_rag --use_schema --use_protocol)
  fi

  "${CMD[@]}"
done < <(jq -r '.[]
                | select(.user_case_seed_query != null)
                | [.id, .source, .core_issue, .user_case_seed_query]
                | @tsv' "$DATA")

echo "Outputs saved to ./output/${MODEL_KEY}"