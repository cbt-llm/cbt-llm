from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

_models = {}


def _get_mpnet():
    if "mpnet" not in _models:
        _models["mpnet"] = SentenceTransformer("all-mpnet-base-v2")
    return _models["mpnet"]


def _get_sapbert():
    if "sapbert" not in _models:
        tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        model.eval()
        _models["sapbert"] = (tokenizer, model)
    return _models["sapbert"]


def _get_bioreddit():
    if "bioreddit" not in _models:
        tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/BioRedditBERT-uncased")
        model = AutoModel.from_pretrained("cambridgeltl/BioRedditBERT-uncased")
        model.eval()
        _models["bioreddit"] = (tokenizer, model)
    return _models["bioreddit"]


def _get_mentalbert():
    if "mentalbert" not in _models:
        tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")
        model = AutoModel.from_pretrained("mental/mental-bert-base-uncased")
        model.eval()
        _models["mentalbert"] = (tokenizer, model)
    return _models["mentalbert"]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
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


def top_k_snomed(driver, query_embedding, embedding_field, k=5):
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

    with driver.session() as session:
        results = session.run(cypher, {"query": query_embedding, "k": k}).data()

    if not results:
        return [{"code": None, "term": "No matches found", "relations": [], "score": None}]

    return results


def retrieve_snomed_matches(driver, text, mode="mpnet", k=5):
    if mode not in EMBEDDING_MODES:
        raise ValueError(f"Unsupported embedding mode: {mode}")

    embed_fn = EMBEDDING_MODES[mode]["embed_fn"]
    embedding_field = EMBEDDING_MODES[mode]["field"]

    query_vec = embed_fn(text)

    return top_k_snomed(driver, query_vec, embedding_field, k=k)
