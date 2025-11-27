from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-mpnet-base-v2") #same as node embedding

def embed_query(text):
    vec = model.encode(text)
    return vec.tolist()

# def top_k_snomed(driver, query_embedding, k=5):
#     cypher = """
#     WITH $query AS queryVec
#     MATCH (n:Concept)
#     WHERE n.embedding IS NOT NULL
#     WITH n, gds.similarity.cosine(n.embedding, queryVec) AS score
#     RETURN n.code AS code,
#        n.term AS term,
#        [(n)-[r]->(m) | {type: type(r), target: m.code}] AS relations,
#        score
#     ORDER BY score DESC
#     LIMIT $k
#     """
#     params = {"query": query_embedding, "k": k}

#     with driver.session() as session:
#         return session.run(cypher, parameters=params).data()

def top_k_snomed(driver, query_embedding, k=5, threshold=0.3):
    cypher = """
    WITH $query AS queryVec
    MATCH (n:Concept)
    WHERE n.embedding IS NOT NULL
    WITH n, gds.similarity.cosine(n.embedding, queryVec) AS score
    RETURN n.code AS code,
           n.term AS term,
           [(n)-[r]->(m) | {type: type(r), target: m.code}] AS relations,
           score
    ORDER BY score DESC
    LIMIT $k
    """
    params = {"query": query_embedding, "k": k}
    
    with driver.session() as session:
        results = session.run(cypher, params).data()
        if not results or results[0]['score'] < threshold:
            return [{"code": None, "term": "No matches found", "relations": [], "score": None}]
        
        return results



def retrieve_snomed_matches(driver, user_text, k=5):
    q_emb = embed_query(user_text)
    return top_k_snomed(driver, q_emb, k)

