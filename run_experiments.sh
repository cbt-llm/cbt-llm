#!/usr/bin/env bash
set -e

# Usage: ./run_experiments.sh [baseline|cbt] [gemma|mistral|deepseek|gpt]
# Example: ./run_experiments.sh cbt mistral

MODE="$1"
MODEL_KEY="$2"

if [[ -z "$MODE" || -z "$MODEL_KEY" ]]; then
  echo "Usage: ./run_experiments.sh [baseline|cbt] [gemma|mistral|deepseek|gpt]"
  exit 1
fi

if [[ "$MODE" != "baseline" && "$MODE" != "cbt" && "$MODE" != "cbt_mcot" ]]; then
  echo "Invalid mode: $MODE"
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

OUTDIR="output/${MODEL_KEY}"
mkdir -p "$OUTDIR"

TURNS=10
K=5

seeds=(
  "So I'm in school, and every few weeks we have, like, papers to write, and I know, I know it. I know I can do it. It's just actually sitting down and doing it. Well, I get kind of annoyed with it, and then I'm like, well, you know, I've already, you know, it's already night time. I'll try and try to do it tomorrow. And then, you know, either tomorrow doesn't happen or the same thing happens, and it's just. And then I end up, you know, rushing right at the end of, you know, due tomorrow, and I'm getting all these five pages done. It's really hard for me.",
  "Last week I was on my way to class and my girlfriend actually called me up saying that I forgot to take out the trash. And I kind of started thinking, wow, like, I forgot to take out the trash. Something so simple, a simple chore. All I had to do was take out the trash today and I forgot to do it. And then I started to think, well, I can't do something as simple as that. Why is she still with me? Or, and why, why can't I do that? Why can't I do a simple task? Am I good enough for anyone? Am I good enough for her? Kind of starting to feel a little fearful that she would just up and leave me.",
  "Recently, I'm just feeling down a lot and I can't really, like, snap out of it. Like, sometimes if it's happened before, I can just, like, get myself back on track, but recently I just find it really difficult to do."
  # "I feel calm most of the time, but sometimes when small things pile up it feels like nothing is going the way it should, and I end up blowing up."
  # "I keep overthinking everything at work, and I'm scared I'm going to get fired even though nobody said anything."
  # "I feel lonely even when I'm around people, and then I get annoyed at myself for feeling that way."
  # "I don’t want to be a burden on other people. So I never share anything and pretend I’m fine and smile."
  # "Sometimes I can’t help but feel this void despite being with people who are close friends and family. I don't know if I am being ungrateful, but I don't feel like I have anyone to talk to when things get hard. I don't belong anywhere. I just don't want to wake up ever again."
)

core_issue=(
  "academic and educational concerns",
  "relationships (romantic, family, friendships)",
  "self-esteem and confidence issues"
)

for i in "${!seeds[@]}"; do
  run=$((i+1))
  seed="${seeds[$i]}"

  echo "Seed $run"

  CMD=(
    python -m src.cbt_llm.multiturn_convo
    --therapist_model "$MODEL"
    --therapist_mode "$MODE"
    --turns "$TURNS"
    --k "$K"
    --seed "$seed"
    --core_issue "$core_issue"
    --transcript_json "${OUTDIR}/${MODE}_transcript_${run}.json"
  )

  if [[ "$MODE" == "cbt" || "$MODE" == "cbt_mcot" ]]; then
    CMD+=(--use_rag --use_schema --use_protocol)
  fi

  "${CMD[@]}"
done


echo "Outputs saved to ./output/${MODEL_KEY}"