import numpy as np
from sentence_transformers import CrossEncoder

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

LABEL_CONTRADICTION = 0
LABEL_ENTAILMENT = 1
LABEL_NEUTRAL = 2

nli_model = CrossEncoder(NLI_MODEL_NAME)


def build_hypothesis(term, relations):
    base = f"This person has {term.lower()}"

    interprets = [
        r["target"]
        for r in relations
        if r.get("type") == "INTERPRETS"
    ]

    if interprets:
        joined = ", ".join(interprets).lower()
        return f"{base}, which involves {joined}."

    return base + "."

def nli_filter(user_text, findings, neutral_threshold=0.5):

    if not findings:
        return []

    pairs = []
    for f in findings:
        hypothesis = build_hypothesis(f["term"], f["relations"])
        pairs.append((user_text, hypothesis))

    scores = nli_model.predict(pairs)

    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    concepts = set()
    

    for i, f in enumerate(findings):

        entail = probs[i][LABEL_ENTAILMENT]
        neutral = probs[i][LABEL_NEUTRAL]
        contra = probs[i][LABEL_CONTRADICTION]

        # if entail >= neutral and entail >= contra:
        #     keep = True
        # elif neutral >= contra:
        #     keep = neutral >= neutral_threshold
        # else:
        #     keep = False

        # if keep:
        concepts.add(f["term"])

    return {"concepts": [{"term": t} for t in concepts]}


if __name__ == "__main__":

    from neo4j import GraphDatabase
    from cbt_llm.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    from cbt_llm.retrieve_snomed import retrieve_snomed_matches

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    query = "I’m calm most of the times but sometimes I end up blowing up."

    print("\nUser Query:", query)

    findings = retrieve_snomed_matches(driver, query, mode="mpnet", k=5)

    print(findings)

    filtered = nli_filter(query, findings)
    print("------NLI Retrivals-------")
    print(filtered)

    # for c in filtered["concepts"]:
    #     print(c["term"])

    driver.close()