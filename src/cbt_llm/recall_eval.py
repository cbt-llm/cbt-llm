import csv
from neo4j import GraphDatabase
from retrieve_snomed import retrieve_snomed_matches
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


EVAL_DATA = [
    (
        "I’ve been feeling constantly overwhelmed and on edge since everything changed so suddenly, and even small daily tasks now feel hard to cope with.",
        "Stress and adjustment reaction"
    ),
    (
        "I feel completely drained and worn down, as if I have nothing left to give, and even getting through the day feels exhausting.",
        "Physical and emotional exhaustion state"
    ),
    (
        "I feel intense fear and uneasiness when I’m away from the people I’m close to, and it’s hard for me to relax or focus until I know they’re okay.",
        "Separation anxiety"
    ),
    (
        "I’m struggling to adapt to recent changes in my life, and I feel unsettled and unsure of myself in ways I didn’t expect.",
        "Adjustment reaction of adult life"
    ),
    (
        "I keep going over the same thoughts again and again, unable to let them go, even when I know it’s not helping me feel better.",
        "Rumination disorder"
    ),
    (
        "I’ve been feeling persistently low and weighed down since the changes in my life, and the sadness hasn’t lifted even as time has passed.",
        "Prolonged depressive adjustment reaction"
    ),
    (
        "I’ve been feeling deeply low for a long time now, with little motivation or hope, and it’s hard to remember what it feels like to be myself again.",
        "Chronic bipolar I disorder, most recent episode depressed"
    ),
    (
        "I feel blocked and stuck when it comes to my studies or work, even though I want to move forward, and it leaves me feeling frustrated and discouraged.",
        "Specific academic or work inhibition"
    ),
    (
        "I feel like I’m not performing to my potential academically, even though I put in effort, and it leaves me feeling disappointed and unsure of my abilities.",
        "Academic underachievement disorder"
    ),
    (
        "I feel constantly overwhelmed by the responsibility of caring for someone else, and it’s becoming harder to manage my own needs without feeling guilty or exhausted.",
        "Carer stress syndrome"
    ),
]

EMBEDDING_MODES = ["mpnet", "sapbert", "bioreddit", "mentalbert"]
K_VALUES = [1, 3, 5]



def norm(text):
    if not text:
        return ""
    text = text.lower().strip()
    for suffix in [
        " (disorder)",
        " (finding)",
        " (situation)",
        " (event)",
        " (procedure)",
    ]:
        text = text.replace(suffix, "")
    return text


def recall_at_k(retrieved_terms, target, k):
    return int(norm(target) in [norm(t) for t in retrieved_terms[:k]])


def main():
    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    
    recall_csv = open("recall_summary.csv", "w", newline="", encoding="utf-8")
    recall_writer = csv.writer(recall_csv)
    recall_writer.writerow(["Embedding", "Recall@1", "Recall@3", "Recall@5"])

    qual_csv = open("qualitative_examples.csv", "w", newline="", encoding="utf-8")
    qual_writer = csv.writer(qual_csv)
    qual_writer.writerow([
        "Embedding",
        "Query",
        "Ground Truth Disorder",
        "Rank",
        "Retrieved Concept",
        "Is Correct"
    ])

    print("\n==============================")
    print(" SNOMED RETRIEVAL EVALUATION ")
    print("==============================")

    for mode in EMBEDDING_MODES:
        hits = {k: 0 for k in K_VALUES}

        print(f"\nEmbedding Model: {mode.upper()}")
        print("--------------------------------------------")

        for query, target in EVAL_DATA:
            results = retrieve_snomed_matches(
                driver, query, mode=mode, k=max(K_VALUES)
            )

            retrieved_terms = [
                r["term"] for r in results if r["term"] != "No matches found"
            ]

            print(f"\nQuery: {query}")
            print(f"Ground Truth: {target}")
            print("Retrieved:")

            for rank, term in enumerate(retrieved_terms, start=1):
                correct = norm(term) == norm(target)
                marker = "" if correct else ""
                print(f"  {rank}. {term} {marker}")

                
                qual_writer.writerow([
                    mode,
                    query,
                    target,
                    rank,
                    term,
                    "YES" if correct else "NO"
                ])

            
            for k in K_VALUES:
                hits[k] += recall_at_k(retrieved_terms, target, k)

        
        recall_writer.writerow([
            mode.upper(),
            f"{hits[1] / len(EVAL_DATA):.2f}",
            f"{hits[3] / len(EVAL_DATA):.2f}",
            f"{hits[5] / len(EVAL_DATA):.2f}",
        ])

        print("\nRecall Summary:")
        for k in K_VALUES:
            print(f"  Recall@{k}: {hits[k] / len(EVAL_DATA):.2f}")

    driver.close()
    recall_csv.close()
    qual_csv.close()

    print("\n==============================")
    print(" Tables generated successfully ")
    print("==============================")
    print("  • recall_summary.csv")
    print("  • qualitative_examples.csv")


if __name__ == "__main__":
    main()
