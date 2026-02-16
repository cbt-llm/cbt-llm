from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm.auto import tqdm
import numpy as np
from huggingface_hub import login




driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)


# mpnet_model= SentenceTransformer("all-mpnet-base-v2")

# sapbert_tokenizer = AutoTokenizer.from_pretrained(
#     "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
# )
# sapbert_model = AutoModel.from_pretrained(
#     "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
# )
# sapbert_model.eval()


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


def fetch_nodes(tx):
    query = """
    MATCH (n:Concept)
    RETURN n.code AS code, n.term AS term, n.synonyms AS synonyms
    """
    return list(tx.run(query))


# def store_embeddings(tx, code, vec_mpnet, vec_sapbert):
#     query = """
#     MATCH (n:Concept {code: $code})
#     SET n.embedding_mpnet = $mpnet,
#         n.embedding_sapbert = $sapbert
#     """
#     tx.run(query, code=code, mpnet=vec_mpnet, sapbert=vec_sapbert)


def store_embeddings(tx, code, bioreddit_vec, mentalbert_vec):
    query = """
    MATCH (n:Concept {code: $code})
    SET n.embedding_bioreddit = $bio,
        n.embedding_mentalbert = $mental
    """
    tx.run(
        query,
        code=code,
        bio=bioreddit_vec,
        mental=mentalbert_vec
    )


# def sapbert_embed(texts, max_length=25):
#     """ Returns CLS embeddings of a list of strings """

#     encoded = sapbert_tokenizer.batch_encode_plus(
#         texts,
#         padding="max_length",
#         truncation=True,
#         max_length=max_length,
#         return_tensors="pt"
#     )

    

#     with torch.no_grad():
#         output = sapbert_model(**encoded)[0][:, 0, :]   # CLS token

#     return output.numpy()


# def main():
#     print("Fetching nodes...")
#     with driver.session() as session:
#         nodes = session.execute_read(fetch_nodes)

#     print("Generating & storing embeddings (MPNet + SapBERT)...")
#     for row in nodes:
#         term = row["term"] or ""
#         synonyms = row["synonyms"] or []
#         combined_text = " ".join([term] + synonyms)

#         # MPNet embedding
#         mpnet_vec = mpnet_model.encode(combined_text).tolist()

#         # SapBERT embedding
#         sap_vec = sapbert_embed([combined_text])[0].tolist()

#         with driver.session() as session:
#             session.execute_write(
#                 store_embeddings,
#                 row["code"],
#                 mpnet_vec,
#                 sap_vec
#             )

#     print("Done! Both embeddings stored.")



# if __name__ == "__main__":
#     main()

def cls_embed(texts, tokenizer, model, max_length=32):
    encoded = tokenizer.batch_encode_plus(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model(**encoded).last_hidden_state[:, 0, :]

    return output.numpy()


# -------- Main -------- #

def main():
    print("Fetching SNOMED nodes...")
    with driver.session() as session:
        nodes = session.execute_read(fetch_nodes)

    print("Generating BioRedditBERT + MentalBERT embeddings...")
    for row in tqdm(nodes):
        term = row["term"] or ""
        synonyms = row["synonyms"] or []
        text = " ".join([term] + synonyms)
        
        bio_vec = cls_embed(
            [text], bioreddit_tokenizer, bioreddit_model
        )[0].tolist()

        mental_vec = cls_embed(
            [text], mentalbert_tokenizer, mentalbert_model
        )[0].tolist()

        with driver.session() as session:
            session.execute_write(
                store_embeddings,
                row["code"],
                bio_vec,
                mental_vec
            )

    print("Done! New embeddings stored safely.")


if __name__ == "__main__":
    main()