from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from cbt_llm.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from tqdm.auto import tqdm
import torch
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
)

mpnet = SentenceTransformer("all-mpnet-base-v2")
def load_hf_model(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.eval()
    return tok, mdl

sapbert_tok, sapbert_mdl = load_hf_model("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
bioreddit_tok, bioreddit_mdl = load_hf_model("cambridgeltl/BioRedditBERT-uncased")
mentalbert_tok, mentalbert_mdl = load_hf_model("mental/mental-bert-base-uncased")

def fetch_nodes(tx):
    q = """
    MATCH (n:Concept)
    RETURN n.code AS code, n.term AS term
    """
    return list(tx.run(q))

def cls_embed(text: str, tok, mdl, max_length=32):
    enc = tok(
        [text],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        vec = mdl(**enc).last_hidden_state[:, 0, :]  # CLS
    return vec.detach().cpu().numpy()[0].tolist()

def store_embeddings(tx, code: str, embeddings: dict):
    """
    embeddings keys must match Neo4j property names:
      embedding_mpnet, embedding_sapbert, embedding_bioreddit, embedding_mentalbert
    """
    q = """
    MATCH (n:Concept {code: $code})
    SET n.embedding_sapbert = $embedding_sapbert,
        n.embedding_bioreddit = $embedding_bioreddit,
        n.embedding_mentalbert = $embedding_mentalbert
    """
    tx.run(q, code=code, **embeddings)

HF_EMBEDDERS = {
    "embedding_sapbert": (sapbert_tok, sapbert_mdl),
    "embedding_bioreddit": (bioreddit_tok, bioreddit_mdl),
    "embedding_mentalbert": (mentalbert_tok, mentalbert_mdl),
}

def main():
    print("Fetching SNOMED nodes...")
    with driver.session() as s:
        nodes = s.execute_read(fetch_nodes)

    print("Generating embeddings (3 HF CLS models)...")
    for row in tqdm(nodes):
        code = row["code"]
        text = row.get("term") or ""

        embeddings = {
            "embedding_mpnet": mpnet.encode(text).tolist()
        }
        embeddings = {}

        for prop, (tok, mdl) in HF_EMBEDDERS.items():
            embeddings[prop] = cls_embed(text, tok, mdl)

        with driver.session() as s:
            s.execute_write(store_embeddings, code, embeddings)

    print("Done! All embeddings stored.")


if __name__ == "__main__":
    main()