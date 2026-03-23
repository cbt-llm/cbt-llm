import os
import json
import pandas as pd
import glob

HUMAN_DIR = "human_eval"
RESPONSE_DIR = "response_eval"
OUTPUT_DIR = "aligned"

os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_files = glob.glob(os.path.join(HUMAN_DIR, "*.csv"))

print("CSV FILES FOUND:", csv_files)

for csv_path in csv_files:

    filename = os.path.basename(csv_path).replace(".csv", "")
    json_path = os.path.join(RESPONSE_DIR, filename + ".json")

    print(f"\nProcessing: {filename}")

    if not os.path.exists(json_path):
        print("Missing JSON")
        continue


    df = pd.read_csv(csv_path)

    rating_cols = [c for c in df.columns if "b_1" in c]

    print(f"Detected {len(rating_cols)} turns")


    df = df[df[rating_cols].notna().any(axis=1)]

    df = df.tail(4).reset_index(drop=True)

    if len(df) != 4:
        print("ERROR: Expected 4 reviewers, got:", len(df))

    with open(json_path) as f:
        js = json.load(f)

    transcript = js["transcript"]
    n = min(len(rating_cols), len(transcript))

    aligned = {
        "metadata": filename,
        "turns": []
    }

    for i in range(n):

        col = rating_cols[i]
        ratings = []

        for r in range(4):
            val = df.iloc[r][col]
            ratings.append(str(val).strip())

        llm = transcript[i].get("llm_response", {})
        used = []

        if llm.get("reasoning"):
            used = llm["reasoning"].get("retrieved_concepts_used", [])

        elif llm.get("mcot_candidates"):
            protocol = llm.get("protocol_used")
            candidate = llm["mcot_candidates"].get(protocol, {})
            reasoning = candidate.get("reasoning")

            if reasoning:
                used = reasoning.get("retrieved_concepts_used", [])

        aligned["turns"].append({
            "turn": i,
            "ratings": ratings,
            "used_concepts": used
        })

    out_path = os.path.join(OUTPUT_DIR, f"{filename}.json")

    with open(out_path, "w") as f:
        json.dump(aligned, f, indent=2)

    print(f"saved: {out_path}")