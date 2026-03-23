import os
import json
import glob
import pandas as pd

ALIGNED_DIR = "aligned"

SNOMED = {"entailment", "neutral", "contradiction"}
COGNITIVE = {"triggers", "automatic_thoughts", "emotions", "behaviors"}

def normalize_rating(r):
    if not isinstance(r, str):
        return ""

    r = r.strip().lower()

    if "inppropriate" in r:
        r = "inappropriate"

    return r



def is_appropriate_label(r):
    r = normalize_rating(r)

    if r == "":
        return None

    if r in {"appropriate", "very appropriate"}:
        return True

    return False


rows = []

files = glob.glob(os.path.join(ALIGNED_DIR, "*.json"))

print("FILES:", files)

for path in files:

    with open(path) as f:
        js = json.load(f)

    name = js["metadata"]
    turns = js["turns"]

    total_ratings = 0
    positive_ratings = 0

    snomed_count = 0
    cognitive_count = 0
    total_concepts = 0

    for t in turns:

        for r in t["ratings"]:
            val = is_appropriate_label(r)

            if val is None:
                continue

            total_ratings += 1

            if val:
                positive_ratings += 1

        for c in t["used_concepts"]:
            label = str(c.get("label", "")).lower()

            if label in SNOMED:
                snomed_count += 1
                total_concepts += 1

            elif label in COGNITIVE:
                cognitive_count += 1
                total_concepts += 1


    appr_pct = (positive_ratings / total_ratings) * 100 if total_ratings else 0

    if total_concepts > 0:
        snomed_pct = (snomed_count / total_concepts) * 100
        cognitive_pct = (cognitive_count / total_concepts) * 100
    else:
        snomed_pct = 0
        cognitive_pct = 0

    rows.append({
        "Model": name,
        "Appropriateness (%)": round(appr_pct, 1),
        "SNOMED CT (%)": round(snomed_pct, 1),
        "Cognitive Model (%)": round(cognitive_pct, 1)
    })


df = pd.DataFrame(rows).sort_values("Model")

print(df.to_string(index=False))