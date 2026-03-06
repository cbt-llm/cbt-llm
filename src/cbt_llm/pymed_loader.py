import pandas as pd
from pymedtermino.snomedct import SNOMEDCT

# Root for all Mental state, behavior and/or psychosocial function finding (finding)
# Reference: https://docs.snomed.org/snomed-ct-specifications/snomed-ct-editorial-guide/readme/authoring/domain-specific-modeling/clinical-finding-and-disorder 
ROOTS = [
    384821006,  # Mental state, behavior and/or psychosocial function finding
]


def extract_snomed_relationships(roots):

    nodes = {}
    relationships = []

    for root_id in roots:

        root = SNOMEDCT[root_id]

        for concept in root.descendants_no_double():

            term = concept.term.lower()

            # remove disorders
            if "(disorder)" in term:
                continue

            # keep only findings
            if "(finding)" not in term:
                continue

            code = str(concept.code)
            nodes[code] = concept.term

    # second pass to create relationships
    for code in nodes:

        concept = SNOMEDCT[int(code)]

        for parent in concept.parents:

            parent_code = str(parent.code)

            if parent_code in nodes:
                relationships.append({
                    "source": code,
                    "relation": "IS_A",
                    "target": parent_code
                })

        try:
            for target in concept.interprets:

                target_code = str(target.code)

                relationships.append({
                    "source": code,
                    "relation": "INTERPRETS",
                    "target": target_code
                })

        except AttributeError:
            pass

    node_rows = [{"code": c, "term": t} for c, t in nodes.items()]

    return node_rows, relationships


if __name__ == "__main__":

    nodes, rels = extract_snomed_relationships(ROOTS)

    pd.DataFrame(nodes).to_csv("mental_nodes.csv", index=False)
    pd.DataFrame(rels).to_csv("mental_relationships.csv", index=False)

    print("Concepts:", len(nodes))
    print("Relationships:", len(rels))