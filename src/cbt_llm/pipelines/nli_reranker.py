import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

from cbt_llm.config import OUTPUT_NEO4J_DIR

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# Label indices for cross-encoder/nli-deberta-v3-small
LABEL_CONTRADICTION = 0
LABEL_ENTAILMENT = 1
LABEL_NEUTRAL = 2


def _build_hypothesis(finding_term, interprets_targets):
    """Build hypothesis from finding term and its INTERPRETS relation targets."""
    base = f"This person has {finding_term.lower()}"
    if interprets_targets:
        joined = ", ".join(t.lower() for t in interprets_targets)
        return f"{base}, which involves {joined}."
    return f"{base}."


def run_nli_reranker(
    input_file="snomed_turn_results.csv",
    output_file="nli_reranked_results.csv",
    json_file="nli_findings.json",
    embedding_filter="mpnet",
    neutral_threshold=0.5,
):
    input_path = Path(OUTPUT_NEO4J_DIR) / input_file
    output_path = Path(OUTPUT_NEO4J_DIR) / output_file
    json_path = Path(OUTPUT_NEO4J_DIR) / json_file

    df = pd.read_csv(input_path)
    df = df[df["Embedding"] == embedding_filter].copy()
    df = df[df["SNOMED Term"].notna()].copy()

    # Collect INTERPRETS targets per finding key
    interprets_by_finding = defaultdict(list)
    for _, row in df.iterrows():
        if row.get("Relation Type") == "INTERPRETS" and pd.notna(row.get("Relation Target Term")):
            key = (int(row["Turn"]), row["User Text"], row["SNOMED Term"], row["Code"], row["Score"])
            interprets_by_finding[key].append(row["Relation Target Term"])

    # Deduplicate to one row per finding
    findings = (
        df[["Turn", "User Text", "SNOMED Term", "Code", "Score"]]
        .drop_duplicates()
        .sort_values(["Turn", "Score"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # Build (premise, hypothesis) pairs
    pairs = []
    for _, row in findings.iterrows():
        key = (int(row["Turn"]), row["User Text"], row["SNOMED Term"], row["Code"], row["Score"])
        targets = interprets_by_finding.get(key, [])
        hypothesis = _build_hypothesis(row["SNOMED Term"], targets)
        pairs.append((row["User Text"], hypothesis))

    print(f"Loading NLI model: {NLI_MODEL_NAME}")
    nli_model = CrossEncoder(NLI_MODEL_NAME)

    print(f"Running NLI on {len(pairs)} findings...")
    raw_scores = nli_model.predict(pairs)  # shape: (n, 3) logits

    # Softmax to convert logits to probabilities
    exp_scores = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    kept = 0
    findings_by_turn = defaultdict(list)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Turn",
            "User Text",
            "SNOMED Term",
            "Code",
            "Retrieval Score",
            "Hypothesis",
            "NLI Label",
            "Entailment Score",
            "Neutral Score",
            "Contradiction Score",
            "Decision",
        ])

        for i, (_, row) in enumerate(findings.iterrows()):
            hypothesis = pairs[i][1]
            entailment_score = float(probs[i][LABEL_ENTAILMENT])
            neutral_score = float(probs[i][LABEL_NEUTRAL])
            contradiction_score = float(probs[i][LABEL_CONTRADICTION])

            if entailment_score >= neutral_score and entailment_score >= contradiction_score:
                label = "ENTAILMENT"
                decision = "KEEP"
            elif neutral_score >= contradiction_score:
                label = "NEUTRAL"
                decision = "KEEP" if neutral_score >= neutral_threshold else "DROP"
            else:
                label = "CONTRADICTION"
                decision = "DROP"

            if decision == "KEEP":
                kept += 1
                findings_by_turn[int(row["Turn"])].append(row["SNOMED Term"])

            writer.writerow([
                row["Turn"],
                row["User Text"],
                row["SNOMED Term"],
                row["Code"],
                row["Score"],
                hypothesis,
                label,
                round(entailment_score, 4),
                round(neutral_score, 4),
                round(contradiction_score, 4),
                decision,
            ])

    json_output = {str(turn): terms for turn, terms in sorted(findings_by_turn.items())}
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(json_output, jf, indent=2)

    print(f"\nNLI reranking complete.")
    print(f"  Total findings : {len(findings)}")
    print(f"  Kept           : {kept}")
    print(f"  Dropped        : {len(findings) - kept}")
    # print(f"  CSV saved      : {output_path}")
    print(f"  JSON saved     : {json_path}\n")
