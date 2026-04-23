import argparse, glob, json, os, re, sys
import ollama


THERAPIST_BASELINE_PROMPT = """
In 2-4 sentences, respond to the user's message.
""".strip()


def generate_for_transcript(input_path, output_path, model_name):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_meta = data.get("metadata", {})
    baseline_turns = []
    for turn in sorted(data.get("transcript", []), key=lambda t: t.get("turn", 0)):
        patient_query = turn.get("patient", {}).get("query", "").strip()
        if not patient_query:
            continue
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": THERAPIST_BASELINE_PROMPT},
                {"role": "user", "content": patient_query},
            ],
            options={"temperature": 0.7},
        )
        baseline_turns.append({
            "turn": turn["turn"],
            "patient_query": patient_query,
            "baseline_response": response["message"]["content"].strip(),
        })

    output = {
        "metadata": {
            "llm_response": model_name,
            "patient_model": source_meta.get("patient_model"),
            "mode": "baseline",
            "turns": source_meta.get("turns"),
            "seed": source_meta.get("seed"),
            "core_issue": source_meta.get("core_issue"),
            "intervention_flags": {
                "use_schema": False,
                "use_rag": False,
                "use_protocol": False,
            },
        },
        "baseline": baseline_turns,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def baseline_filename(input_name):
    """Map cbt_mcot_transcript_N.json -> baseline_transcript_N.json."""
    match = re.search(r"(\d+)\.json$", input_name)
    if not match:
        return f"baseline_{input_name}"
    return f"baseline_transcript_{match.group(1)}.json"


def main():
    ap = argparse.ArgumentParser(description="Generate baseline therapist responses")
    ap.add_argument("--input_dir", required=True, help="MCOT transcript dir")
    ap.add_argument("--output_dir", required=True, help="Where to write baselines")
    ap.add_argument("--model", required=True,
                    help="Ollama model name, e.g. mistral:7b, gemma3:12b, gpt-oss:20b")
    ap.add_argument("--overwrite", action="store_true",
                    help="Regenerate even if baseline file exists")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    if not input_files:
        sys.exit(f"No JSON files found in {args.input_dir}")

    print(f"Generating baselines for {len(input_files)} transcripts using {args.model}\n")

    for input_path in input_files:
        output_name = baseline_filename(os.path.basename(input_path))
        output_path = os.path.join(args.output_dir, output_name)
        if os.path.exists(output_path) and not args.overwrite:
            print(f"  skip (exists): {output_name}")
            continue
        generate_for_transcript(input_path, output_path, args.model)
        print(f"  done: {output_name}")


if __name__ == "__main__":
    main()