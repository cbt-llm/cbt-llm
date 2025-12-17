mkdir -p output

seeds=(
  "I keep overthinking everything at work, and I'm scared I'm going to get fired even though nobody said anything."
  "I feel lonely even when I'm around people, and then I get annoyed at myself for feeling that way."
  "I get a tight chest before meetings and I avoid speaking, then I regret it all day."
  "I messed up one small thing yesterday and now I can't stop replaying it like it proves I'm not good enough."
  "My sleep is messed up because my brain won't shut off at night, and I wake up tired and irritated."

)

for i in "${!seeds[@]}"; do
  run=$((i+1))

  ---- BASELINE (no RAG, no schema) ----
  python -m cbt_llm.multiturn_convo \
    --therapist_model "gemma2:9b" \
    --patient_model "mistral:7b-instruct" \
    --turns 6 \
    --k 5 \
    --seed "${seeds[$i]}" \
    --transcript_json "output/baseline_transcript_${run}.json" \
    --retrieval_json "output/baseline_retrieval_${run}.json" \
    --prompt_trace_json "output/baseline_prompt_trace_${run}.json"

  # ---- RAG (+ optional schema) ----
  python -m cbt_llm.multiturn_convo \
    --therapist_model "gemma2:9b" \
    --patient_model "mistral:7b-instruct" \
    --turns 6 \
    --k 5 \
    --use_rag \
    --use_schema \
    --seed "${seeds[$i]}" \
    --transcript_json "output/rag_transcript_${run}.json" \
    --retrieval_json "output/rag_retrieval_${run}.json" \
    --prompt_trace_json "output/rag_prompt_trace_${run}.json"


python src/cbt_llm/new_vadar.py output --out output/vader_outputs

done
