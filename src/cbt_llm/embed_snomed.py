from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD



driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)


model = SentenceTransformer("all-mpnet-base-v2")


def fetch_nodes(tx):
    query = """
    MATCH (n:Concept)
    RETURN n.code AS code, n.term AS term, n.synonyms AS synonyms
    """
    return list(tx.run(query))


def store_embedding(tx, code, embedding):
    query = """
    MATCH (n:Concept {code: $code})
    SET n.embedding = $vec
    """
    tx.run(query, code=code, vec=embedding)


def main():
    print("Fetching nodes...")
    with driver.session() as session:
        nodes = session.execute_read(fetch_nodes)

    print("Generating & storing embeddings...")
    for row in nodes:
        term = row["term"] or ""
        synonyms = row["synonyms"] or []
        combined_text = " ".join([term] + synonyms)

        vec = model.encode(combined_text).tolist()

        with driver.session() as session:
            session.execute_write(store_embedding, row["code"], vec)

    print("Done! Embeddings stored.")


if __name__ == "__main__":
    main()

