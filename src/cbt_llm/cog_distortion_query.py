from neo4j import GraphDatabase

class SnomedQuery:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def fetch_related_concepts(self, thoughts, emotions, behaviors, limit=10):
        """
        Query SNOMED graph using schema components:
        - automatic_thoughts
        - emotion
        - behavior

        This DOES NOT diagnose. It only retrieves related SNOMED concepts.
        """

        keywords = []
        for lst in [thoughts, emotions, behaviors]:
            if isinstance(lst, list):
                keywords.extend(lst)
            elif isinstance(lst, str) and lst.strip():
                keywords.append(lst)

        keywords = list({kw.lower().strip() for kw in keywords if kw and len(kw) > 2})

        cypher = """
        UNWIND $keywords AS kw
        CALL db.index.fulltext.queryNodes(
          'conceptTermIndex',
          kw
        ) YIELD node, score
        RETURN node.code AS code,
               node.term AS term,
               score
        ORDER BY score DESC
        LIMIT $limit;
        """

        with self.driver.session() as session:
            result = session.run(cypher, keywords=keywords, limit=limit)
            return [record.data() for record in result]
