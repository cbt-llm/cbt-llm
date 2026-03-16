import json
import csv
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("output")

SCHEMA_LABELS = {
    "triggers",
    "automatic_thoughts",
    "emotions",
    "behaviors",
}

SNOMED_LABELS = {
    "entailment",
    "neutral",
    "contradiction",
}

INTERVENTIONS = [
    "cot",
    "validate_and_reflect",
    "socratic_questioning",
    "cognitive_restructuring",
]

MODE_MAP = {
    "cbt": "CBT-CoT",
    "cbt_mcot": "CBT-MCoT",
}

counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))


def classify_label(label):

    if label in SCHEMA_LABELS:
        return "schema"

    if label in SNOMED_LABELS:
        return "snomed"

    if label == "unknown":
        return "unknown"

    return "other"


def process_reasoning(reasoning):

    if not reasoning:
        return None

    concepts = reasoning.get("retrieved_concepts_used", [])

    if not concepts:
        return None

    result = {
        "schema": 0,
        "snomed": 0,
        "other": 0,
        "unknown": 0,
    }

    for c in concepts:

        label = c.get("label")

        cat = classify_label(label)

        result[cat] += 1

    return result


def process_file(path):

    data = json.loads(path.read_text())

    model = data["metadata"]["llm_response"]

    mode_raw = data["metadata"]["mode"]

    if mode_raw not in MODE_MAP:
        return

    mode = MODE_MAP[mode_raw]

    transcript = data["transcript"]

    for turn in transcript:

        llm = turn["llm_response"]

        # ---- MCOT ----
        if "protocol_used" in llm:

            candidates = llm.get("mcot_candidates", {})

            for protocol, candidate in candidates.items():

                reasoning = candidate.get("reasoning")

                stats = process_reasoning(reasoning)

                if stats is None:
                    counts[model][mode][protocol]["null_reasoning"] += 1
                    continue

                for k, v in stats.items():
                    counts[model][mode][protocol][k] += v

            continue

        # ---- COT ----
        else:

            protocol = "cot"

            reasoning = llm.get("reasoning")

        stats = process_reasoning(reasoning)

        if stats is None:

            counts[model][mode][protocol]["null_reasoning"] += 1

            continue

        for k, v in stats.items():

            counts[model][mode][protocol][k] += v


def scan_outputs():

    for model_dir in OUTPUT_DIR.iterdir():

        if not model_dir.is_dir():
            continue

        for f in model_dir.glob("*.json"):

            process_file(f)


def write_csv():

    with open("concept_usage.csv", "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "response_model",
            "mode",
            "intervention",
            "schema_concepts",
            "snomed_concepts",
            "other_concepts",
            "unknown_labels",
            "null_reasoning"
        ])

        for model in sorted(counts):

            for mode in ["CBT-CoT", "CBT-MCoT"]:

                if mode == "CBT-CoT":
                    interventions = ["cot"]
                else:
                    interventions = [
                        "validate_and_reflect",
                        "socratic_questioning",
                        "cognitive_restructuring",
                    ]

                for intervention in interventions:

                    row = counts[model][mode][intervention]

                    writer.writerow([
                        model,
                        mode,
                        "CBT-CoT" if intervention == "cot" else intervention,
                        row["schema"],
                        row["snomed"],
                        row["other"],
                        row["unknown"],
                        row["null_reasoning"],
                    ])


if __name__ == "__main__":

    scan_outputs()

    write_csv()

    print("Saved concept_usage.csv")