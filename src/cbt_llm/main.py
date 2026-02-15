<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
from pipelines.neo4j_pipeline import run_neo4j_pipeline
from pipelines.llm_pipeline import run_all_models
from patient_queries import PATIENT_TURNS

def main():
    print("Choose pipeline:")
    print("1 → Neo4j SNOMED retrieval")
    print("2 → LLM semantic extraction")

    choice = input("Enter choice: ")

    if choice == "1":
        # Neo4j: call pipeline one turn at a time
        for i, turn in enumerate(PATIENT_TURNS, start=1):
            print(f"\n========== Neo4j Turn {i} ==========\n")
            run_neo4j_pipeline([turn])  # still passes a list with one turn

    elif choice == "2":
        # LLM: call pipeline one turn at a time
        for i, turn in enumerate(PATIENT_TURNS, start=1):
            print(f"\n========== LLM Turn {i} ==========\n")
            run_all_models(turn)  # expects a single string

    else:
        print("Invalid choice.")
<<<<<<< Updated upstream

    print("\n\nResults saved to snomed_turn_results.csv\n")
=======
>>>>>>> Stashed changes


if __name__ == "__main__":
    main()
