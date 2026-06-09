#!/usr/bin/env bash
set -e

# Usage: ./run_experiments_ablation.sh [gemma|mistral|deepseek|gpt]
# Example: ./run_experiments_ablation.sh gemma
#
# Runs three ablation variants per case study:
#   no_rag      — removes SNOMED retrieval + NLI verification
#   no_schema   — removes cognitive model (schema extraction)
#   no_protocol — removes CBT principles/protocols

MODEL_KEY="$1"

if [[ -z "$MODEL_KEY" ]]; then
  echo "Usage: ./run_experiments_ablation.sh [gemma|mistral|deepseek|gpt]"
  exit 1
fi

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

command -v jq >/dev/null || { echo "jq is required (brew install jq / apt-get install jq)"; exit 1; }

DATA="data/processed/ablation_case_studies.json"
[[ -f "$DATA" ]] || { echo "Dataset not found: $DATA"; exit 1; }

OUTDIR="output/${MODEL_KEY}/ablation"
mkdir -p "$OUTDIR"

TURNS=10
K=5

VARIANTS=("no_rag" "no_schema")
# VARIANTS=("no_rag" "no_schema" "no_protocol")


while IFS=$'\t' read -r id source issue seed; do
  echo "=== Case $id ($source) — core issue: $issue ==="

  for VARIANT in "${VARIANTS[@]}"; do
    echo "  Running ablation: $VARIANT"

    python -m cbt_llm.ablation_convo \
      --therapist_model "$MODEL" \
      --ablation_variant "$VARIANT" \
      --turns "$TURNS" \
      --k "$K" \
      --seed "$seed" \
      --core_issue "$issue" \
      --transcript_json "${OUTDIR}/${VARIANT}_transcript_${id}.json"

    echo "  Saved: ${OUTDIR}/${VARIANT}_transcript_${id}.json"
  done

done < <(jq -r '.[]
                | select(.user_case_seed_query != null)
                | [.id, .source, .core_issue, .user_case_seed_query]
                | @tsv' "$DATA")

echo "Ablation outputs saved to ./${OUTDIR}"
