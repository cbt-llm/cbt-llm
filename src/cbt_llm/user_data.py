import pandas as pd
import re
from pathlib import Path

def extract_statements_from_file(file_path, column_name="client_statement"):
    """
    Reads a CSV file and extracts the client_statement column.
    Ensures proper parsing and returns full sentences (one per row).
    Cleans bracket tags like [PATIENT] at start.
    Removes newline characters and normalizes whitespace.
    """

    # Read CSV robustly
    df = pd.read_csv(
        file_path,
        sep=",",
        quotechar='"',
        engine="python",
        encoding="utf-8"
    )

    # Normalize column names
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    column_name = column_name.strip().lower().replace(" ", "_")

    if column_name not in df.columns:
        raise ValueError(
            f"Column '{column_name}' not found in {file_path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Drop empty rows
    statements = df[column_name].dropna().astype(str).tolist()

    cleaned_statements = []
    for s in statements:
        # Remove leading bracket tag like [PATIENT]
        cleaned = re.sub(r"^\[\w+\]\s*", "", s)

        # Remove newline characters
        cleaned = cleaned.replace("\n", " ")

        # Normalize multiple spaces into single space
        cleaned = re.sub(r"\s+", " ", cleaned)

        cleaned_statements.append(cleaned.strip())

    return cleaned_statements


def load_all_patient_turns(csv_files, column_name="client_statement"):
    """
    Combines all statements from multiple CSV files into a single list.
    Each row remains one full sentence.
    """
    all_turns = []

    for file in csv_files:
        print(f"Loading: {file}")
        statements = extract_statements_from_file(file, column_name)
        all_turns.extend(statements)

    print(f"\nTotal patient turns loaded: {len(all_turns)}")
    return all_turns