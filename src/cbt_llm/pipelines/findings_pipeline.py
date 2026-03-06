"""
findings_pipeline.py
--------------------
Single entry point for SNOMED retrieval + NLI re-ranking.

Usage:
    from cbt_llm.pipelines.findings_pipeline import FindingsPipeline

    pipeline = FindingsPipeline(driver)
    findings = pipeline.get_findings("I don't know why I keep blowing up at people.")
    # {
    #   "entailment":    ["Unable to control anger (finding)", ...],
    #   "neutral":       ["Tends to allow anger to build up (finding)"],
    #   "contradiction": ["Able to control anger (finding)"]
    # }
"""

import numpy as np
from sentence_transformers import CrossEncoder

from cbt_llm.retrieve_snomed import retrieve_snomed_matches

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

LABEL_CONTRADICTION = 0
LABEL_ENTAILMENT = 1
LABEL_NEUTRAL = 2


class FindingsPipeline:
    def __init__(self, driver, k=5, neutral_threshold=0.5):
        self.driver = driver
        self.k = k
        self.neutral_threshold = neutral_threshold
        print(f"Loading NLI model: {NLI_MODEL_NAME}")
        self._nli = CrossEncoder(NLI_MODEL_NAME)

    def get_findings(self, query):
        """
        Run SNOMED retrieval + NLI re-ranking for a single patient query.

        Returns:
            dict with keys "entailment", "neutral", "contradiction",
            each mapping to a list of SNOMED term strings.
        """
        results = retrieve_snomed_matches(self.driver, query, mode="mpnet", k=self.k)

        if not results or results[0]["term"] == "No matches found":
            return {"entailment": [], "neutral": [], "contradiction": []}

        # Hypothesis = finding term only
        pairs = [
            (query, f"This person has {r['term'].lower()}.")
            for r in results
        ]

        raw_scores = self._nli.predict(pairs)
        exp_scores = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        output = {"entailment": [], "neutral": [], "contradiction": []}

        for i, r in enumerate(results):
            entailment_score = float(probs[i][LABEL_ENTAILMENT])
            neutral_score = float(probs[i][LABEL_NEUTRAL])
            contradiction_score = float(probs[i][LABEL_CONTRADICTION])

            if entailment_score >= neutral_score and entailment_score >= contradiction_score:
                output["entailment"].append(r["term"])
            elif neutral_score >= contradiction_score:
                if neutral_score >= self.neutral_threshold:
                    output["neutral"].append(r["term"])
                else:
                    output["contradiction"].append(r["term"])
            else:
                output["contradiction"].append(r["term"])

        return output

def main():
    results = FindingsPipeline.get_findings("I keep blowing up.")
    print(results)

if __name__ == "__main__":
    main()