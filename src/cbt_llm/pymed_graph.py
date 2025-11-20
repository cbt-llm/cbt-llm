from neo4j import GraphDatabase

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
            UNWIND $rows AS row
            MATCH (a:Concept {code: row.source})
            MATCH (b:Concept {code: row.target})
            MERGE (a)-[r:REL {type: row.relation}]->(b)
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
    from cbt_llm.pymed_loader import extract_snomed_relationships
    from cbt_llm.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

    nodes, rels = extract_snomed_relationships(74732009)

    ingestor = LoadSnomedGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    ingestor.load(nodes, rels)
    ingestor.close()

    print("Neo4j graph created")
