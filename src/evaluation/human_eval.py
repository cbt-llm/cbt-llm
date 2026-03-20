import pandas as pd
import numpy as np

# -----------------------------
# Config
# -----------------------------
FILES = {
    "mcot-mistral-1.csv": ("Mistral-7B", "MCoT"),
    "mcot-gemma-2.csv": ("Gemma3-4B", "MCoT"),
    "cot-deepseek-3.csv": ("DeepSeek-R1", "CoT"),
    "cot-gpt-4.csv": ("GPT-OSS-20B", "CoT"),
}

LIKERT_MAP = {
    "Very inappropriate": 1,
    "Inappropriate": 2,
    "Neutral": 3,
    "Appropriate": 4,
    "Very appropriate": 5
}

PROTOCOL_MAP = {
    "Socratic Questioning": "SQ",
    "Validation & Reflection": "V&R",
    "Cognitive Restructuring": "CR",
    "Neither/Other": "None"
}


# -----------------------------
# Helpers
# -----------------------------
def extract_protocols(cell):
    if pd.isna(cell):
        return []
    
    vals = str(cell).split(",")
    return [PROTOCOL_MAP[v.strip()] for v in vals if v.strip() in PROTOCOL_MAP]


# -----------------------------
# Main
# -----------------------------
results = []

for file, (model, mode) in FILES.items():
    df = pd.read_csv(f"human_eval/{file}")
    
    # auto-detect relevant columns
    protocol_cols = [c for c in df.columns if c.endswith("a")]
    score_cols = [c for c in df.columns if "b_1" in c]
    
    protocol_counts = {"V&R": 0, "SQ": 0, "CR": 0, "None": 0}
    total_selections = 0  # 🔥 FIXED normalization
    scores = []
    agreements = []
    ambiguities = []
    
    for _, row in df.iterrows():
        
        for a_col, b_col in zip(protocol_cols, score_cols):
            
            protocols = extract_protocols(row[a_col])
            
            if not protocols:
                continue
            
            # -----------------------------
            # Protocol counting (FIXED)
            # -----------------------------
            for p in protocols:
                protocol_counts[p] += 1
                total_selections += 1
            
            # -----------------------------
            # Likert score
            # -----------------------------
            score_text = str(row[b_col]).strip()
            if score_text in LIKERT_MAP:
                scores.append(LIKERT_MAP[score_text])
            
            # -----------------------------
            # Agreement (simple proxy)
            # -----------------------------
            agreements.append(1 if len(set(protocols)) <= 2 else 0)
            
            # -----------------------------
            # Ambiguity
            # -----------------------------
            ambiguities.append(len(set(protocols)))
    
    if total_selections == 0:
        raise ValueError(f"No protocol selections found in {file}")
    
    # -----------------------------
    # Normalize correctly
    # -----------------------------
    protocol_pct = {
        k: round((v / total_selections) * 100, 1)
        for k, v in protocol_counts.items()
    }
    
    results.append({
        "Model": model,
        "Mode": mode,
        "V&R": protocol_pct["V&R"],
        "SQ": protocol_pct["SQ"],
        "CR": protocol_pct["CR"],
        "None": protocol_pct["None"],
        "Mean": round(np.mean(scores), 2),
        "Var": round(np.var(scores), 2),
        "Agree": round(np.mean(agreements), 2),
        "Ambig": round(np.mean(ambiguities) / 4, 2)
    })


# -----------------------------
# Final table
# -----------------------------
final_df = pd.DataFrame(results)

print("\nFinal Table:\n")
print(final_df.to_string(index=False))

