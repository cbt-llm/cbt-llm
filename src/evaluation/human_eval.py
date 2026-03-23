import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import krippendorff

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

PROTOCOL_LABELS = ["V&R", "SQ", "CR", "None"]

def extract_protocols(cell):
    if pd.isna(cell):
        return []
    vals = str(cell).split(",")
    return [PROTOCOL_MAP[v.strip()] for v in vals if v.strip() in PROTOCOL_MAP]


def compute_alpha(data, level):
    try:
        max_len = max(len(x) for x in data)
        padded = [x + [np.nan] * (max_len - len(x)) for x in data]
        arr = np.array(padded).T
        return krippendorff.alpha(reliability_data=arr, level_of_measurement=level)
    except Exception as e:
        print(f"Alpha computation failed: {e}")
        return None


results = []
model_scores = {}

likert_ratings = []      # for Krippendorff alpha (ordinal)
protocol_matrix = []     # for Krippendorff alpha (nominal)

for file, (model, mode) in FILES.items():
    df = pd.read_csv(f"human_eval/{file}")
    
    protocol_cols = [c for c in df.columns if c.endswith("a")]
    score_cols = [c for c in df.columns if "b_1" in c]
    
    protocol_counts = {k: 0 for k in PROTOCOL_LABELS}
    total_selections = 0
    scores = []
    ambiguities = []
    
    for _, row in df.iterrows():
        
        item_scores = []
        item_protocols = []
        
        for a_col, b_col in zip(protocol_cols, score_cols):
            
            protocols = extract_protocols(row[a_col])
            
            if not protocols:
                continue
            
           
            for p in protocols:
                protocol_counts[p] += 1
                total_selections += 1
            
            
            score_text = str(row[b_col]).strip()
            if score_text in LIKERT_MAP:
                score_val = LIKERT_MAP[score_text]
                scores.append(score_val)
                item_scores.append(score_val)
          
            item_protocols.append(protocols[0])
        
     
        if len(item_scores) > 1:
            likert_ratings.append(item_scores)
        
        if len(item_protocols) > 1:
            proto_numeric = [
                PROTOCOL_LABELS.index(p)
                for p in item_protocols if p in PROTOCOL_LABELS
            ]
            protocol_matrix.append(proto_numeric)
        
       
        if item_protocols:
            ambiguities.append(len(set(item_protocols)))
    
    if total_selections == 0:
        raise ValueError(f"No protocol selections found in {file}")
    
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
        "Ambig": round(np.mean(ambiguities) / 4, 2)
    })
    
    model_scores[model] = scores


final_df = pd.DataFrame(results)

print("\nFinal Table:\n")
print(final_df.to_string(index=False))


print("\nStatistical Significance (Mann–Whitney U):\n")

pairs = [
    ("Gemma3-4B", "GPT-OSS-20B"),
    ("Gemma3-4B", "DeepSeek-R1"),
    ("Mistral-7B", "GPT-OSS-20B"),
]

for m1, m2 in pairs:
    if m1 in model_scores and m2 in model_scores:
        stat, p = mannwhitneyu(
            model_scores[m1],
            model_scores[m2],
            alternative='two-sided'
        )
        print(f"{m1} vs {m2}: p = {p:.4f} (n1={len(model_scores[m1])}, n2={len(model_scores[m2])})")


print("\nInter-rater Reliability (Krippendorff’s Alpha):\n")

if likert_ratings:
    alpha_likert = compute_alpha(likert_ratings, "ordinal")
    print(f"Effectiveness (ordinal α): {alpha_likert:.4f}")

if protocol_matrix:
    alpha_proto = compute_alpha(protocol_matrix, "nominal")
    print(f"Protocol selection (nominal α): {alpha_proto:.4f}")