from neo4j import GraphDatabase
from pymedtermino.snomedct import SNOMEDCT


ROOTS = [
    384821006  # Mental state, behavior and/or psychosocial function finding
]


def extract_snomed_relationships(roots):

    nodes = {}
    relationships = []

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

    for code in list(nodes.keys()):

        concept = SNOMEDCT[int(code)]

        for parent in concept.parents:

            parent_code = str(parent.code)

            if parent_code not in nodes:
                nodes[parent_code] = parent.term

            relationships.append({
                "source": code,
                "relation": "IS_A",
                "target": parent_code
            })

        try:
            for target in concept.interprets:

                target_code = str(target.code)

                if target_code not in nodes:
                    nodes[target_code] = target.term

                relationships.append({
                    "source": code,
                    "relation": "INTERPRETS",
                    "target": target_code
                })

        except AttributeError:
            pass

    node_rows = [{"code": c, "term": t} for c, t in nodes.items()]

    return node_rows, relationships


class LoadSnomedGraph:

    def __init__(self, uri, user, password, batch_size=500):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.batch_size = batch_size

    def setup_constraints(self):
        with self.driver.session() as session:
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (c:Concept) REQUIRE c.code IS UNIQUE
            """)

    def _load_nodes(self, tx, nodes):
        tx.run("""
            UNWIND $rows AS row
            MERGE (c:Concept {code: row.code})
            SET c.term = row.term
        """, rows=nodes)

    def _load_rels(self, tx, rels):
        tx.run("""
            UNWIND $rows AS r
            MATCH (a:Concept {code:r.source})
            MATCH (b:Concept {code:r.target})
            MERGE (a)-[:REL {type:r.relation}]->(b)
        """, rows=rels)

    def load(self, nodes, relationships):

        self.setup_constraints()

        with self.driver.session() as session:

            for i in range(0, len(nodes), self.batch_size):
                session.execute_write(self._load_nodes, nodes[i:i+self.batch_size])

            for i in range(0, len(relationships), self.batch_size):
                session.execute_write(self._load_rels, relationships[i:i+self.batch_size])

    def close(self):
        self.driver.close()


if __name__ == "__main__":
    from cbt_llm.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

    nodes, rels = extract_snomed_relationships(ROOTS)

    print("Concepts:", len(nodes))
    print("Relationships:", len(rels))

    loader = LoadSnomedGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    loader.load(nodes, rels)
    loader.close()

    print("Graph created")