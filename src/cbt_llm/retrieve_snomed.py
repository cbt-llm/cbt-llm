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
    return mpnet_model.encode(text).tolist()

def embed_query_sapbert(text, max_length=256):
    encoded = sapbert_tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = sapbert_model(**encoded)

    vec = mean_pooling(output, encoded["attention_mask"])
    return vec[0].numpy().tolist()


def embed_query_bioreddit(text, max_length=256):
    encoded = bioreddit_tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = bioreddit_model(**encoded)

    vec = mean_pooling(output, encoded["attention_mask"])
    return vec[0].numpy().tolist()


def embed_query_mentalbert(text, max_length=256):
    encoded = mentalbert_tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = mentalbert_model(**encoded)

    vec = mean_pooling(output, encoded["attention_mask"])
    return vec[0].numpy().tolist()





def top_k_snomed(driver, query_embedding, embedding_field, k=5, threshold=0):
    cypher = f"""
    WITH $query AS queryVec
    MATCH (n:Concept)
    WHERE n.{embedding_field} IS NOT NULL
    WITH n, gds.similarity.cosine(n.{embedding_field}, queryVec) AS score
    RETURN 
        n.code AS code,
        n.term AS term,
        [(n)-[r]->(m) | {{type: coalesce(r.type, type(r)), target: m.code}}] AS relations,
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


def retrieve_snomed_matches(driver, text, mode="bioreddit", k=5, threshold=0):
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




