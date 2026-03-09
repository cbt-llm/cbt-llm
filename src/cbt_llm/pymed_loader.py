import pandas as pd
from pymedtermino.snomedct import SNOMEDCT


# Root: Mental state, behavior and/or psychosocial function finding
ROOTS = [
    384821006
]


def extract_snomed_relationships(roots):

    nodes = {}
    relationships = []

    # ---------------------------------
    # Pass 1: collect mental findings
    # ---------------------------------
    for root_id in roots:

        root = SNOMEDCT[root_id]

        for concept in root.descendants_no_double():

            term = concept.term.lower()

            if "(disorder)" in term:
                continue

            if "(finding)" not in term:
                continue

            code = str(concept.code)
            nodes[code] = concept.term

    # ---------------------------------
    # Pass 2: build relationships
    # ---------------------------------
    for code in list(nodes.keys()):

        concept = SNOMEDCT[int(code)]

        # ---------- IS_A ----------
        for parent in concept.parents:

            parent_code = str(parent.code)

            # ensure parent node exists
            if parent_code not in nodes:
                nodes[parent_code] = parent.term

            relationships.append({
                "source": code,
                "relation": "IS_A",
                "target": parent_code
            })

        # ---------- INTERPRETS ----------
        try:
            for target in concept.interprets:

                target_code = str(target.code)

                # ensure interpreted node exists
                if target_code not in nodes:
                    nodes[target_code] = target.term

                relationships.append({
                    "source": code,
                    "relation": "INTERPRETS",
                    "target": target_code
                })

        except AttributeError:
            pass

    # ---------------------------------
    # convert nodes to rows
    # ---------------------------------
    node_rows = [{"code": c, "term": t} for c, t in nodes.items()]

    return node_rows, relationships


if __name__ == "__main__":

    nodes, rels = extract_snomed_relationships(ROOTS)

    pd.DataFrame(nodes).to_csv("mental_nodes.csv", index=False)
    pd.DataFrame(rels).to_csv("mental_relationships.csv", index=False)

    print("Concepts:", len(nodes))
    print("Relationships:", len(rels))

    print(
        "INTERPRETS:",
        sum(1 for r in rels if r["relation"] == "INTERPRETS")
    )