from neo4j import GraphDatabase
from cbt_llm.retrieve_snomed import retrieve_snomed_matches
from cbt_llm.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OUTPUT_NEO4J_DIR
import csv
from pathlib import Path

EMBEDDING_MODES = ["mpnet", "sapbert", "bioreddit", "mentalbert"]


def fetch_term(driver, code):
    if code is None:
        return None
    with driver.session() as session:
        result = session.run(
            "MATCH (n:Concept {code: $code}) RETURN n.term AS term LIMIT 1",
            code=code
        ).single()
    return result["term"] if result else None


def run_neo4j_pipeline(patient_turns, output_file="snomed_turn_results.csv"):
    output_path = Path(OUTPUT_NEO4J_DIR) / output_file
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow([
            "Embedding",
            "Turn",
            "User Text",
            "SNOMED Term",
            "Code",
            "Score",
            "Relation Type",
            "Relation Target Code",
            "Relation Target Term"
        ])

        for i, turn in enumerate(patient_turns, start=1):

            for mode in EMBEDDING_MODES:
                results = retrieve_snomed_matches(driver, turn, mode=mode, k=5)

                for r in results:

                    if r["term"] == "No matches found":
                        writer.writerow([mode, i, turn, None, None, None, None, None, None])
                        continue

                    code = r["code"]
                    term = r["term"]
                    score = round(r["score"], 4)

                    if r["relations"]:
                        for rel in r["relations"]:
                            rel_type = rel["type"]
                            target_code = rel["target"]
                            target_term = fetch_term(driver, target_code)

                            writer.writerow([
                                mode,
                                i, turn,
                                term, code, score,
                                rel_type, target_code, target_term
                            ])
                    else:
                        writer.writerow([
                            mode,
                            i, turn,
                            term, code, score,
                            None, None, None
                        ])

    driver.close()

    print(f"\nResults saved to {output_path}\n")
