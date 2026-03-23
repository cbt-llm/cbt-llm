import os
import json
import glob
import pandas as pd
import numpy as np
from collections import defaultdict

INPUT_DIR = "./output"
USAGE_CSV = "concept_usage.csv"

MODEL_MAP = {
    "gpt": "gpt-oss:20b",
    "gemma": "gemma3-12b",
    "deepseek": "deepseek-r1:8b",
    "mistral": "mistral-7b",
}


def normalize_label(label):
    if not label:
        return None
    label = label.lower()

    if "entail" in label:
        return "entail"
    elif "neutral" in label:
        return "neutral"
    elif "contradict" in label:
        return "contradict"
    else:
        return None


def compute_entropy(probs):
    probs = [p for p in probs if p > 0]
    if len(probs) == 0:
        return 0.0

    entropy = -sum(p * np.log(p) for p in probs)

    # normalize by log(3) since 3 categories (E,N,C)
    return entropy / np.log(3)

def init_counts():
    return {
        "retrieved": {"entail": 0, "neutral": 0, "contradict": 0},
        "used": {"entail": 0, "neutral": 0, "contradict": 0},
        "total_retrieved": 0,
        "total_used": 0
    }

usage_df = pd.read_csv(USAGE_CSV)

grounding = {}
for _, row in usage_df.iterrows():
    key = (row["response_model"], row["mode"])
    grounding[key] = {
        "other": row["other_concepts"],
        "unknown": row["unknown_labels"],
        "null": row["null_reasoning"],
    }

print(f"Loaded grounding entries: {len(grounding)}")

files = glob.glob(os.path.join(INPUT_DIR, "**", "*_transcript_*.json"), recursive=True)
print(f"Found {len(files)} transcript files")


data = defaultdict(init_counts)

for file_path in files:

    if "baseline" in file_path:
        continue

    with open(file_path, "r") as f:
        js = json.load(f)

    folder_model = file_path.split(os.sep)[-2]
    model = MODEL_MAP.get(folder_model, folder_model)

    filename = os.path.basename(file_path)

    if "cbt_mcot" in filename:
        mode = "cbt_mcot"
        mode_key = "CBT-MCoT"
    elif "cbt_" in filename:
        mode = "cbt"
        mode_key = "CBT-CoT"
    else:
        continue

    key = (model, mode)

    for turn in js.get("transcript", []):

        retrieval = turn.get("patient", {}).get("retrieval") or {}

        ent = len(retrieval.get("entailment", []))
        neu = len(retrieval.get("neutral", []))
        con = len(retrieval.get("contradiction", []))

        data[key]["retrieved"]["entail"] += ent
        data[key]["retrieved"]["neutral"] += neu
        data[key]["retrieved"]["contradict"] += con
        data[key]["total_retrieved"] += (ent + neu + con)

        llm = turn.get("llm_response", {})
        used_concepts = []

        if llm.get("reasoning"):
            used_concepts = llm["reasoning"].get("retrieved_concepts_used", [])

        elif llm.get("mcot_candidates"):
            protocol = llm.get("protocol_used")
            candidate = llm["mcot_candidates"].get(protocol, {})
            reasoning = candidate.get("reasoning")

            if reasoning:
                used_concepts = reasoning.get("retrieved_concepts_used", [])

        for c in used_concepts:
            label = normalize_label(c.get("label"))
            if label:
                data[key]["used"][label] += 1
                data[key]["total_used"] += 1


rows = []

for (model, mode), stats in data.items():

    total_r = stats["total_retrieved"]
    total_u = stats["total_used"]

    if total_r == 0:
        continue

    r_ent = stats["retrieved"]["entail"] / total_r
    r_neu = stats["retrieved"]["neutral"] / total_r
    r_con = stats["retrieved"]["contradict"] / total_r

    if total_u > 0:
        u_ent = stats["used"]["entail"] / total_u
        u_neu = stats["used"]["neutral"] / total_u
        u_con = stats["used"]["contradict"] / total_u
    else:
        u_ent = u_neu = u_con = 0

    mode_key = "CBT-MCoT" if mode == "cbt_mcot" else "CBT-CoT"
    g = grounding.get((model, mode_key), {"other": 0, "unknown": 0, "null": 0})

    other = g.get("other", 0)
    unknown = g.get("unknown", 0)
    null = g.get("null", 0)

    total_extended = total_u + other + unknown + null

    if total_extended > 0:
        u_other_null = (other + unknown + null) / total_extended
    else:
        u_other_null = 0

    delta_ent = (u_ent - r_ent) * 100

    
    H_u = compute_entropy([u_ent, u_neu, u_con])

    rows.append({
        "Model": model,
        "Mode": mode,
        "Retrieved Entail (%)": round(r_ent * 100, 1),
        "Retrieved Neutral (%)": round(r_neu * 100, 1),
        "Retrieved Contradict (%)": round(r_con * 100, 1),
        "Used Entail (%)": round(u_ent * 100, 1),
        "Used Neutral (%)": round(u_neu * 100, 1),
        "Used Contradict (%)": round(u_con * 100, 1),
        "Used (O+N) (%)": round(u_other_null * 100, 1),
        "H_u (Entropy)": round(H_u, 3)
    })

df = pd.DataFrame(rows)

if not df.empty:
    df = df.sort_values(by=["Model", "Mode"])

print("\n=== FINAL ENTROPY TABLE (CLEAN) ===\n")
print(df.to_string(index=False))

output_path = os.path.join(INPUT_DIR, "concept_uncertainty_table.csv")
df.to_csv(output_path, index=False)

print(f"\nSaved to: {output_path}")


# retrieval performance 

# are we a

# - how likely i retrieced from snomed ct * its usage across entailment

# how likely retrieved concepts are identified as 

# - does the concepts improve performance


# all responses that came out mcot, cot-> how many were apt and above (V, CR, SQ)
# -> is the response better because of snomed or 
# -> label response
# -> similarity to snomed vs. response 


# Table baseline -> snomed ct or cognitive model 
# if reponse is apt and higher with atleast then compare with -> snomed concepts (how much %)


# Ethical considerations

# manas's mental health paper: ethical consideration -> no personal transcripts
# -> synthetic 


# no such dataset exist for V, SQ, CR, so we use simulated conversation 

