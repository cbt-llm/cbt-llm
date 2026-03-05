# main.py

from cbt_llm.pipelines.neo4j_pipeline import run_neo4j_pipeline
# from cbt_llm.pipelines.llm_pipeline import run_all_models
# from cbt_llm.pipelines.claude_pipeline import run_claude_model
from cbt_llm.pipelines.nli_reranker import run_nli_reranker
from cbt_llm.user_data import load_all_patient_turns
from cbt_llm.patient_queries import PATIENT_TURNS
import glob
import os


# Path to all CSV files in the folder
_data_dir = os.path.join(os.path.dirname(__file__), "cbt_user_data")
csv_files = glob.glob(os.path.join(_data_dir, "*.csv"))

#Load and clean the patient turns
PATIENT_TURNS = load_all_patient_turns(csv_files, column_name="client_statement")
# print(PATIENT_TURNS[:5])


def main():
    print("Choose pipeline:")
    print("1 → Neo4j SNOMED retrieval")
    # print("2 → LLM semantic extraction")
    # print("3 → Claude semantic extraction")
    print("4 → NLI re-ranking (from snomed_turn_results.csv)")

    choice = input("Enter choice: ")

    if choice == "1":
        run_neo4j_pipeline(PATIENT_TURNS)
    # elif choice == "2":
    #     run_all_models(PATIENT_TURNS)
    # elif choice == "3":
    #     run_claude_model(PATIENT_TURNS)
    elif choice == "4":
        run_nli_reranker()
    else:
        print("Invalid choice.")

    print("\n\nResults saved to snomed_turn_results.csv\n")


if __name__ == "__main__":
    main()