import os
from pathlib import Path
from dotenv import load_dotenv

from cbt_llm.config import SNOMEDCT_DIR, SNOMEDCT_CORE_FILE

import sys
pymed_path = Path(SNOMEDCT_DIR) / "build" / "lib"
sys.path.insert(0, str(pymed_path))

from pymedtermino.snomedct import SNOMEDCT


def extract_snomed_relationships(root_id):
    root = SNOMEDCT[root_id]

    nodes = {}
    relationships = []

    for concept in root.descendants_no_double():

        nodes[str(concept.code)] = concept.term

        for parent in concept.parents:
            relationships.append({
                "source": str(concept.code),
                "relation": "IS_A",
                "target": str(parent.code)
            })

        for rel_type in concept.relations:
            try:
                targets = getattr(concept, rel_type)
            except AttributeError:
                continue

            if not targets:
                continue

            for target in targets:
                relationships.append({
                    "source": str(concept.code),
                    "relation": rel_type,
                    "target": str(target.code)
                })

    node_rows = [
        {"code": code, "term": term}
        for code, term in nodes.items()
    ]

    return node_rows, relationships


if __name__ == "__main__":
    print("Filtering for SNOMED mental health data")

    # starting root at Mental disorders and mapping all its children/relationships
    nodes, rels = extract_snomed_relationships(74732009)

    print("Concepts:", len(nodes))
    print("Relations:", len(rels))
