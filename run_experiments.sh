#!/usr/bin/env bash
set -e

# Usage: ./run_experiments.sh [baseline|cbt|cbt_mcot] [gemma|mistral|deepseek|gpt] [realcbt|esconv] [random|thoughtful]
# Example: ./run_experiments.sh cbt_mcot mistral realcbt
# Example: ./run_experiments.sh cbt_mcot mistral esconv

MODE="$1"
MODEL_KEY="$2"
DATASET="$3"          # realcbt or esconv
DISTRACTION_TYPE="$4" # random | thoughtful (optional, kept for ablation)

if [[ -z "$MODE" || -z "$MODEL_KEY" || -z "$DATASET" ]]; then
  echo "Usage: ./run_experiments.sh [baseline|cbt|cbt_mcot] [gemma|mistral|deepseek|gpt] [realcbt|esconv] [random|thoughtful]"
  exit 1
fi

if [[ "$DATASET" != "realcbt" && "$DATASET" != "esconv" ]]; then
  echo "Invalid dataset: $DATASET. Must be 'realcbt' or 'esconv'."
  exit 1
fi

if [[ "$MODE" != "baseline" && "$MODE" != "cbt" && "$MODE" != "cbt_mcot" ]]; then
  echo "Invalid mode: $MODE"
  exit 1
fi

if [[ -n "$DISTRACTION_TYPE" && "$DISTRACTION_TYPE" != "random" && "$DISTRACTION_TYPE" != "thoughtful" ]]; then
  echo "Invalid distraction type: $DISTRACTION_TYPE. Must be 'random' or 'thoughtful'."
  exit 1
fi

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

OUTDIR="output/${MODEL_KEY}_${DATASET}${DISTRACTION_TYPE:+_${DISTRACTION_TYPE}}"
mkdir -p "$OUTDIR"

TURNS=10
K=5

# ---------------------------------------------------------------------------
# Seed / core_issue arrays — commented out; patient turns now come from
# real transcripts. Kept here for reference and synthetic ablation experiments.
# ---------------------------------------------------------------------------
# seeds=(
#   "I feel calm most of the time, but sometimes when small things pile up it feels like nothing is going the way it should, and I end up blowing up."
#   # "I know I'm a burden on other people, so I keep things to myself and put on a smile, even when I'm not okay."
#   # "I keep overthinking everything at work, and I'm scared I'm going to get fired even though nobody said anything."
#   # "I feel lonely even when I'm around people, and then I get annoyed at myself for feeling that way."
#   # "I don't know why I can't just start things. I have so much to do but I just sit there and nothing happens."
#   # "Every time something goes well I immediately start waiting for it to fall apart. I can't just enjoy things."
#   # "I feel like I always have to be the one holding everything together, and if I stop, everything will collapse."
#   # "I get really anxious before social situations, even ones I've done a hundred times. My mind just goes blank."
#   # "I keep replaying conversations in my head and thinking about what I should have said differently."
#   # "I feel disconnected from everything lately, like I'm just going through the motions but nothing feels real."
# )
#
# core_issues=(
#   "anger and emotional dysregulation"
#   # "self-esteem and confidence issues"
#   # "anxiety and fear"
#   # "loneliness and social connection"
#   # "procrastination and avoidance"
#   # "anxiety and fear"
#   # "self-esteem and confidence issues"
#   # "anxiety and fear"
#   # "rumination and overthinking"
#   # "dissociation and low mood"
# )

# ---------------------------------------------------------------------------
# Distraction schedules — kept for synthetic ablation but not injected when
# running real transcripts (client turns are fixed from the source file).
# ---------------------------------------------------------------------------
# Completely off-topic, lighthearted tangents
random_distractions=(
  "Anyway, I spent like thirty minutes watching videos of parrots swearing at people and honestly that might've been the highlight of my week. Also randomly went down a Reddit rabbit hole about relationship drama — thank god I'm not in one. Life is so much more peaceful alone. I'll just die alone."
  "There's this squirrel that keeps coming to my window sill and I've been feeding him peanuts. I named him Gerald. Gerald came back today and I think he brought a friend."
  "I've been trying to get into meditation but every time I close my eyes I just start making a grocery list. Ended up buying seventeen things yesterday including oat milk, which I don't even like."
)

# Emotionally adjacent but drifting
thoughtful_distractions=(
  "Actually, I have a friend who gets way more angry than me — she threw something at a wall last week. I wonder if it runs in families. My dad had a really bad temper too and I used to just stay out of his way."
  "My friend has been leaning on me a lot lately for help with his project. I feel guilty even thinking about saying no. It's like everyone needs something from me and I don't know how to push back."
  "A friend told me she went through something similar last year and therapy helped her a lot. I keep thinking maybe I should try it, but I'm not sure I'm ready to really go there yet. It feels like a big step."
)

# Distraction injection — commented out for real-transcript runs
# DISTRACTIONS_JSON=""
# if [[ "$DISTRACTION_TYPE" == "random" ]]; then
#   DISTRACTIONS_JSON=$(python3 -c "
# import json, sys
# turns = [3, 5, 8]
# texts = sys.argv[1:]
# print(json.dumps({str(t): txt for t, txt in zip(turns, texts)}))" \
#     "${random_distractions[0]}" "${random_distractions[1]}" "${random_distractions[2]}")
# elif [[ "$DISTRACTION_TYPE" == "thoughtful" ]]; then
#   DISTRACTIONS_JSON=$(python3 -c "
# import json, sys
# turns = [2, 4, 5]
# texts = sys.argv[1:]
# print(json.dumps({str(t): txt for t, txt in zip(turns, texts)}))" \
#     "${thoughtful_distractions[0]}" "${thoughtful_distractions[1]}" "${thoughtful_distractions[2]}")
# fi

# ---------------------------------------------------------------------------
# Dataset loops
# ---------------------------------------------------------------------------

if [[ "$DATASET" == "realcbt" ]]; then

  REALCBT_DIR="$(dirname "$0")/data/raw/realcbt_split"
  mapfile -t client_files < <(ls "${REALCBT_DIR}"/*_client.txt | sort -V)

  for i in "${!client_files[@]}"; do
    run=$((i+1))
    transcript_file="${client_files[$i]}"
    fname=$(basename "$transcript_file" _client.txt)

    echo "Run $run / ${#client_files[@]} | dataset=realcbt | file='$fname'"

    CMD=(
      python -m cbt_llm.multiturn_convo
      --therapist_model "$MODEL"
      --therapist_mode "$MODE"
      --turns "$TURNS"
      --k "$K"
      --transcript_source "$transcript_file"
      # --seed "$seed"        # commented out — turns come from transcript
      # --core_issue "$issue" # commented out — inferred post-session
      --transcript_json "${OUTDIR}/realcbt_${MODE}_${fname}.json"
    )

    if [[ "$MODE" == "cbt" || "$MODE" == "cbt_mcot" ]]; then
      CMD+=(--use_rag --use_schema --use_protocol)
    fi

    "${CMD[@]}"
  done

elif [[ "$DATASET" == "esconv" ]]; then

  ESCONV_JSON="$(dirname "$0")/data/raw/ESConv_client.json"
  ESCONV_COUNT=$(python3 -c "import json; print(len(json.load(open('${ESCONV_JSON}'))))")

  for i in $(seq 0 $((ESCONV_COUNT - 1))); do
    run=$((i+1))

    echo "Run $run / ${ESCONV_COUNT} | dataset=esconv | index=$i"

    CMD=(
      python -m cbt_llm.multiturn_convo
      --therapist_model "$MODEL"
      --therapist_mode "$MODE"
      --turns "$TURNS"
      --k "$K"
      --transcript_source "$ESCONV_JSON"
      --transcript_index "$i"
      # --seed "$seed"        # commented out — turns come from transcript
      # --core_issue "$issue" # commented out — inferred post-session
      --transcript_json "${OUTDIR}/esconv_${MODE}_transcript_${run}.json"
    )

    if [[ "$MODE" == "cbt" || "$MODE" == "cbt_mcot" ]]; then
      CMD+=(--use_rag --use_schema --use_protocol)
    fi

    "${CMD[@]}"
  done

fi

echo "Outputs saved to ./${OUTDIR}"
