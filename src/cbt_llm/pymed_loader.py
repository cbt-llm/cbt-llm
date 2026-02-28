import pandas as pd
from pymedtermino.snomedct import SNOMEDCT

# Root for all Mental state, behavior and/or psychosocial function finding (finding)
# Reference: https://docs.snomed.org/snomed-ct-specifications/snomed-ct-editorial-guide/readme/authoring/domain-specific-modeling/clinical-finding-and-disorder 
MENTAL_ROOT = 384821006 

def extract_snomed_relationships(root_id):
    root = SNOMEDCT[root_id]

    nodes = {}
    relationships = []

    for concept in root.descendants_no_double():
        text = concept.term if concept.term else concept.fsn
        nodes[str(concept.code)] = text

        for parent in concept.parents:
            relationships.append({
                "source": str(concept.code),
                "relation": "IS_A",
                "target": str(parent.code)
            })

        try:
            interpreted_targets = concept.interprets
        except AttributeError:
            interpreted_targets = []

        for target in interpreted_targets:
            relationships.append({
                "source": str(concept.code),
                "relation": "INTERPRETS",
                "target": str(target.code)
            })
            target_text = target.term if target.term else target.fsn
            nodes[str(target.code)] = target_text

    node_rows = [{"code": code, "term": term} for code, term in nodes.items()]
    return node_rows, relationships


if __name__ == "__main__":
    nodes, rels = extract_snomed_relationships(MENTAL_ROOT)

    df_nodes = pd.DataFrame(nodes)
    df_rels = pd.DataFrame(rels)

    df_nodes.to_csv("mental_findings_nodes.csv", index=False)
    df_rels.to_csv("mental_findings_rels.csv", index=False)

    print("Concepts:", len(nodes))
    print("Relationships:", len(rels))