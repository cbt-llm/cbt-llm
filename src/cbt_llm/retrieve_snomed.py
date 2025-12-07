from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

mpnet_model = SentenceTransformer("all-mpnet-base-v2")

sapbert_tokenizer = AutoTokenizer.from_pretrained(
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
)
sapbert_model = AutoModel.from_pretrained(
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
)
sapbert_model.eval()


def embed_query_mpnet(text):
    return mpnet_model.encode(text).tolist()

def embed_query_sapbert(text, max_length=25):
    tokens = sapbert_tokenizer.encode_plus(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    with torch.no_grad():
        out = sapbert_model(**tokens)[0][:, 0, :]   # CLS token

    return out.numpy()[0].tolist()


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

def top_k_snomed(driver, query_embedding, embedding_field, k=5, threshold=0.30):
    cypher = f"""
    WITH $query AS queryVec
    MATCH (n:Concept)
    WHERE n.{embedding_field} IS NOT NULL
    WITH n, gds.similarity.cosine(n.{embedding_field}, queryVec) AS score
    RETURN 
        n.code AS code,
        n.term AS term,
        [(n)-[r]->(m) | {{type: type(r), target: m.code}}] AS relations,
        score
    ORDER BY score DESC
    LIMIT $k
    """

    params = {"query": query_embedding, "k": k}

    with driver.session() as session:
        results = session.run(cypher, params).data()

        if not results or results[0]["score"] < threshold:
            return [{"code": None, "term": "No matches found", "relations": [], "score": None}]

        return results



def retrieve_snomed_matches(driver, text, mode="sapbert", k=5):
    if mode == "mpnet":
        vec = embed_query_mpnet(text)
        field = "embedding_mpnet"
    elif mode == "sapbert":
        vec = embed_query_sapbert(text)
        field = "embedding_sapbert"
    # else:
    #     vec = embed_query_hybrid(text)
    #     field = "embedding_mpnet"   # hybrid still compares to mpnet (average)
    
    return top_k_snomed(driver, vec, field, k)

# def retrieve_snomed_matches(driver, user_text, k=5):
#     q_emb = embed_query(user_text)
#     return top_k_snomed(driver, q_emb, k)

