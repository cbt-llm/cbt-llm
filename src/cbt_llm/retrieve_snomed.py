from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from neo4j import GraphDatabase


mpnet_model = SentenceTransformer("all-mpnet-base-v2")

sapbert_tokenizer = AutoTokenizer.from_pretrained(
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
)
sapbert_model = AutoModel.from_pretrained(
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
)
sapbert_model.eval()

bioreddit_tokenizer = AutoTokenizer.from_pretrained(
    "cambridgeltl/BioRedditBERT-uncased"
)
bioreddit_model = AutoModel.from_pretrained(
    "cambridgeltl/BioRedditBERT-uncased"
)
bioreddit_model.eval()

mentalbert_tokenizer = AutoTokenizer.from_pretrained(
    "mental/mental-bert-base-uncased"
)
mentalbert_model = AutoModel.from_pretrained(
    "mental/mental-bert-base-uncased"
)
mentalbert_model.eval()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # [B, T, H]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)


def embed_query_mpnet(text):
    return _get_mpnet().encode(text).tolist()


def embed_query_sapbert(text, max_length=256):
    tokenizer, model = _get_sapbert()
    encoded = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
    return mean_pooling(output, encoded["attention_mask"])[0].numpy().tolist()


def embed_query_bioreddit(text, max_length=256):
    tokenizer, model = _get_bioreddit()
    encoded = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
    return mean_pooling(output, encoded["attention_mask"])[0].numpy().tolist()


def embed_query_mentalbert(text, max_length=256):
    tokenizer, model = _get_mentalbert()
    encoded = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
    return mean_pooling(output, encoded["attention_mask"])[0].numpy().tolist()


EMBEDDING_MODES = {
    "mpnet": {
        "embed_fn": embed_query_mpnet,
        "field": "embedding_mpnet"
    },
    "sapbert": {
        "embed_fn": embed_query_sapbert,
        "field": "embedding_sapbert"
    },
    "bioreddit": {
        "embed_fn": embed_query_bioreddit,
        "field": "embedding_bioreddit"
    },
    "mentalbert": {
        "embed_fn": embed_query_mentalbert,
        "field": "embedding_mentalbert"
    }
}


def top_k_snomed(driver, query_embedding, embedding_field, k=5, threshold=0.35):
    cypher = f"""
    WITH $query AS queryVec
    MATCH (n:Concept)
    WHERE n.{embedding_field} IS NOT NULL
    WITH n, gds.similarity.cosine(n.{embedding_field}, queryVec) AS score
    WHERE score >= $threshold
    RETURN
        n.code AS code,
        n.term AS term,
        [(n)-[r]->(m) | {{type: type(r), target: m.code}}] AS relations,
        score
    ORDER BY score DESC
    LIMIT $k
    """

    params = {"query": query_embedding, "k": k, "threshold": threshold}

    with driver.session() as session:
        results = session.run(cypher, params).data()

    if not results:
        return [{"code": None, "term": "No matches found", "relations": [], "score": None}]

    return results


def retrieve_snomed_matches(driver, text, mode="mpnet", k=5, threshold=0.35):
    if mode not in EMBEDDING_MODES:
        raise ValueError(f"Unsupported embedding mode: {mode}")

    embed_fn = EMBEDDING_MODES[mode]["embed_fn"]
    embedding_field = EMBEDDING_MODES[mode]["field"]

    query_vec = embed_fn(text)

    return top_k_snomed(
        driver,
        query_vec,
        embedding_field,
        k=k,
        threshold=threshold
    )



if __name__ == "__main__":

    from neo4j import GraphDatabase
    from cbt_llm.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    query = "I keep overthinking everything at work and I'm scared I might get fired."

    print("\nQuery:")
    print(query)
    print("\nTop SNOMED Matches:\n")

    results = retrieve_snomed_matches(
        driver,
        query,
        mode="mpnet",
        k=5,
        threshold=0
    )

    for r in results:
        print(f"TERM  : {r['term']}")
        print(f"CODE  : {r['code']}")
        print(f"SCORE : {r['score']}")

        if r["relations"]:
            print("RELATIONS:")
            for rel in r["relations"]:
                print("  ", rel)

        print("-" * 40)

    driver.close()