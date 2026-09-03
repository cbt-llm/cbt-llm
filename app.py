"""

User interface for live CBT conversations.

Run:
streamlit run cbt-llm/app.py
"""
import os
import json
import re
import logging
import html as html_lib
import base64
from io import BytesIO
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Streamlit's source watcher can probe torch.classes as if it were a normal
# Python package. PyTorch exposes this as a dynamic namespace, which can raise
# a harmless RuntimeError on some Streamlit/Python combinations (notably 3.13).
# Giving the namespace an ordinary empty path prevents that watcher probe from
# trying to instantiate a nonexistent custom class.
try:
    import torch
    torch.classes.__path__ = []
except Exception:
    pass

from sentence_transformers import SentenceTransformer

from cbt_llm.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, ROOT
from cbt_llm.retrieve_snomed import retrieve_snomed_matches

# Load project configuration before importing user_schema: that module creates
# its OpenAI client at import time.
load_dotenv(ROOT / ".env", override=False)
load_dotenv(override=False)

LOGGER = logging.getLogger(__name__)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
SBERT_MODEL = os.getenv("CBT_LENS_SBERT_MODEL", "all-mpnet-base-v2")
MODEL_OPTIONS = ("gemma3:12b", "mistral:7b", "gpt-oss:20b")
CARDS_PER_PAGE = 3

PROTOCOL_PURPOSES = {
    "validation": (
        "Establish accurate understanding of the patient's experience and "
        "psychological safety before cognitive intervention."
    ),
    "socratic": (
        "Facilitate insight through questioning to help patient draw their own "
        "conclusions rather than providing answers directly"
    ),
    "alternative": (
        "Guide the patient toward a more balanced view of their situation by "
        "gently broadening how they interpret their experience"
    ),
}
PROTOCOL_CODES = {"V": "validation", "SQ": "socratic", "AP": "alternative"}
CIRCUMPLEX_CMAP = LinearSegmentedColormap.from_list(
    "truncated_purples", cm.Purples(np.linspace(0.35, 1.0, 256))
)

try:
    # This is the package path used by the experiment runner.
    from user_schema.user_schema import extract_user_schema
except Exception:
    try:
        # Compatibility with repositories that keep the module under cbt_llm.
        from cbt_llm.user_schema import extract_user_schema
    except Exception:
        extract_user_schema = None

USER_SCHEMA_AVAILABLE = extract_user_schema is not None and bool(os.getenv("OPENAI_API_KEY"))


class OllamaChat:
    def __init__(self, model: str, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        num_predict: int,
        top_p: float = 0.9,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "top_p": top_p,
            },
            "stream": False,
        }
        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
        response.raise_for_status()
        content = (response.json().get("message") or {}).get("content", "").strip()
        if not content:
            raise RuntimeError("Ollama returned an empty response.")
        return content


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance for L2-normalized sentence embeddings."""
    return 1.0 - float(np.dot(a, b))


@st.cache_resource(show_spinner=False)
def get_sbert_encoder() -> SentenceTransformer:
    return SentenceTransformer(SBERT_MODEL)


def centroid(vectors: np.ndarray) -> np.ndarray:
    value = np.mean(vectors, axis=0)
    norm = np.linalg.norm(value)
    return value / norm if norm > 0 else value


@st.cache_resource(show_spinner=False)
def get_protocol_mass() -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    encoder = get_sbert_encoder()
    labels = list(PROTOCOL_PURPOSES)
    embeddings = encoder.encode(
        [PROTOCOL_PURPOSES[label] for label in labels],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    masses, purpose_embeddings = {}, {}
    for index, label in enumerate(labels):
        others = [embeddings[j] for j in range(len(labels)) if j != index]
        masses[label] = float(np.mean([cosine_distance(embeddings[index], row) for row in others]))
        purpose_embeddings[label] = embeddings[index]
    return masses, purpose_embeddings


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            return None


def select_protocol_and_classify_concepts(
    llm: OllamaChat,
    user_text: str,
    concepts: Optional[List[Dict[str, Any]]] = None,
    schema: Optional[Dict[str, Any]] = None,
    recent_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Select the protocol before generation and classify retrieved SNOMED relations."""
    concept_terms = [str(item.get("term", "")).strip() for item in (concepts or []) if item.get("term")]
    prompt = f"""
Select the single CBT protocol that should guide the next response.
Return ONLY valid JSON with this schema:
{{
  "selected_protocol": "V" | "SQ" | "AP",
  "snomed_concepts": [{{"term": string, "relation": "entailment" | "contradiction" | "neutral"}}]
}}

Protocol meanings:
V = Validation & Reflection. Select when psychological safety and emotional acknowledgement
must come before examining or reframing a belief.
SQ = Socratic Questioning. Select when a specific assumption can be examined through one
focused, open-ended question that helps the user reach their own conclusion.
AP = Alternative Perspective. Select when a rigid interpretation should be broadened by
introducing a plausible, more balanced interpretation.

Choose the intervention that should shape the response, not merely the tone of the user's message.
Do not select V solely because the user expresses distress. Prefer SQ or AP when the response
should actively examine or broaden an identifiable belief.

Classify every supplied SNOMED concept by whether the USER language entails it, contradicts it,
or is neutral toward it. Preserve each term exactly and do not add concepts.

SNOMED CONCEPTS:
{json.dumps(concept_terms, ensure_ascii=False)}

COGNITIVE MODEL:
{json.dumps(schema or {}, ensure_ascii=False)}

RECENT CONVERSATION:
{json.dumps((recent_history or [])[-4:], ensure_ascii=False)}

CURRENT USER TURN:
{user_text}
""".strip()
    raw = llm.chat(
        [{"role": "system", "content": "You are a precise research evaluator. Output JSON only."},
         {"role": "user", "content": prompt}],
        temperature=0.0,
        num_predict=260,
        top_p=0.2,
    )
    data = _extract_json_object(raw)
    if not data:
        raise RuntimeError("Protocol selector returned invalid JSON.")
    selected = str(data.get("selected_protocol") or "").upper()
    if selected not in PROTOCOL_CODES:
        raise RuntimeError(f"Protocol selector returned an unsupported strategy: {selected!r}")
    returned_labels = {
        str(item.get("term", "")).casefold(): str(item.get("relation", "neutral")).casefold()
        for item in (data.get("snomed_concepts") or [])
        if isinstance(item, dict)
    }
    concept_groups = {"entailment": [], "contradiction": [], "neutral": []}
    for term in concept_terms:
        relation = returned_labels.get(term.casefold(), "neutral")
        relation = relation if relation in concept_groups else "neutral"
        concept_groups[relation].append(term)
    return {
        "selected_protocol": selected,
        "snomed_groups": concept_groups,
    }


BASELINE_SYSTEM_PROMPTS = [
    "Respond naturally as a general-purpose conversational assistant. Be helpful and concise. Do not use hidden CBT, clinical ontology, or cognitive-model guidance.",
    "Give a normal conversational response to the user in 2–4 sentences. Do not apply an explicit therapeutic protocol or structured clinical reasoning framework.",
    "Reply as an ordinary supportive language model using only the visible conversation. Avoid specialized CBT instructions or hidden clinical grounding.",
]


def generate_baseline_replies(
    llm: OllamaChat, visible_history: List[Dict[str, str]], user_text: str
) -> List[str]:
    history = [m for m in visible_history[-6:] if m.get("role") in {"user", "assistant"}]
    replies = []
    for system_prompt in BASELINE_SYSTEM_PROMPTS:
        messages = [{"role": "system", "content": system_prompt}] + history
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != user_text:
            messages.append({"role": "user", "content": user_text})
        replies.append(llm.chat(messages, temperature=0.15, num_predict=140, top_p=0.7))
    return replies


def compute_force_series(contexts: List[Dict[str, Any]]) -> List[Optional[float]]:
    """Apply the supplied F_A definition to every aligned live-session turn."""
    usable_indices = [
        index for index, item in enumerate(contexts)
        if item.get("patient_text") and item.get("therapist_reply")
        and item.get("evaluation", {}).get("protocol_applied")
        and item.get("evaluation", {}).get("selected_protocol") in PROTOCOL_CODES
        and len(item.get("evaluation", {}).get("baseline_replies", [])) == len(BASELINE_SYSTEM_PROMPTS)
    ]
    usable = [contexts[index] for index in usable_indices]
    if not usable:
        return [None] * len(contexts)

    encoder = get_sbert_encoder()
    patient_embeddings = encoder.encode(
        [item["patient_text"] for item in usable], normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    guided_embeddings = encoder.encode(
        [item["therapist_reply"] for item in usable], normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    baseline_embeddings = [
        encoder.encode(
            [item["evaluation"]["baseline_replies"][variant] for item in usable],
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )
        for variant in range(len(BASELINE_SYSTEM_PROMPTS))
    ]
    centroids = [centroid(rows) for rows in baseline_embeddings]
    perturbation = float(np.mean([
        cosine_distance(centroids[i], centroids[j])
        for i, j in combinations(range(len(centroids)), 2)
    ]))
    masses, _ = get_protocol_mass()
    primary_centroid = centroids[0]
    values: List[Optional[float]] = [None] * len(contexts)
    for index, item in enumerate(usable):
        protocol = PROTOCOL_CODES.get(item["evaluation"].get("selected_protocol"))
        mass = masses.get(protocol, 1.0)
        # The supplied script defines shift with this sign; F uses its squared magnitude.
        shift = (
            cosine_distance(guided_embeddings[index], patient_embeddings[index])
            - cosine_distance(baseline_embeddings[0][index], patient_embeddings[index])
        )
        denominator = cosine_distance(guided_embeddings[index], primary_centroid) + perturbation
        values[usable_indices[index]] = (
            100.0 * mass * shift ** 2 / denominator if denominator > 0 else None
        )
    return values


def compute_nclid(contexts: List[Dict[str, Any]], context_window: int = 2) -> Optional[Dict[str, float]]:
    """Compute the supplied dialogue-level nCLiD definition with sentence-BERT."""
    pairs = [item for item in contexts if item.get("patient_text") and item.get("therapist_reply")]
    n = len(pairs)
    if n < 2:
        return None
    encoder = get_sbert_encoder()
    therapist_embeddings = encoder.encode(
        [item["therapist_reply"] for item in pairs], normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    user_embeddings = encoder.encode(
        [item["patient_text"] for item in pairs], normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    local_distances = [
        min(cosine_distance(therapist_embeddings[i], user_embeddings[j])
            for j in range(i, min(i + context_window, n)))
        for i in range(n)
    ]
    uclid = float(np.mean(local_distances))
    pair_sum = sum(
        cosine_distance(therapist_embeddings[i], therapist_embeddings[j])
        + cosine_distance(user_embeddings[i], user_embeddings[j])
        for i, j in combinations(range(n), 2)
    )
    pair_sum += sum(
        cosine_distance(therapist_embeddings[i], user_embeddings[j])
        for i in range(n) for j in range(i, n)
    )
    alpha = (2.0 / (n * (n - 1))) * pair_sum
    if alpha <= 0:
        return None
    return {"N": n, "uCLiD": uclid, "alpha": alpha, "nCLiD": uclid / alpha}


def _nrc_vad_directory() -> Path:
    configured = os.getenv("NRC_VAD_DIR")
    candidates = [
        Path(configured).expanduser() if configured else None,
        ROOT / "external_libs" / "NRC-VAD-Lexicon-v2.1",
        ROOT.parent / "external_libs" / "NRC-VAD-Lexicon-v2.1",
    ]
    for candidate in candidates:
        if candidate and (candidate / "Unigrams").is_dir():
            return candidate
    return candidates[1]


@st.cache_data(show_spinner=False)
def load_nrc_vad(lexicon_dir: str) -> Dict[str, Dict[str, float]]:
    """Load the three NRC-VAD unigram dimensions exactly as in the evaluation script."""
    unigrams_dir = Path(lexicon_dir) / "Unigrams"

    def load_dimension(filename: str) -> Dict[str, float]:
        frame = pd.read_csv(unigrams_dir / filename, sep="\t", header=0)
        frame.columns = [column.strip().lower() for column in frame.columns]
        word_column, score_column = frame.columns[:2]
        frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
        return frame.dropna(subset=[score_column]).set_index(word_column)[score_column].to_dict()

    valence = load_dimension("unigrams-valence-NRC-VAD-Lexicon-v2.1.txt")
    arousal = load_dimension("unigrams-arousal-NRC-VAD-Lexicon-v2.1.txt")
    dominance = load_dimension("unigrams-dominance-NRC-VAD-Lexicon-v2.1.txt")
    return {
        word: {"valence": valence[word], "arousal": arousal[word], "dominance": dominance[word]}
        for word in set(valence) & set(arousal) & set(dominance)
    }


def extract_user_turn_sentiments(
    contexts: List[Dict[str, Any]], vad: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """Apply the supplied unigram matching and cumulative-mean calculation."""
    rows = []
    for turn, item in enumerate(contexts, start=1):
        tokens = re.findall(r"\b[a-z]+\b", str(item.get("patient_text", "")).lower())
        matched = [vad[token] for token in tokens if token in vad]
        if matched:
            rows.append({
                "turn": turn,
                "valence": float(np.mean([entry["valence"] for entry in matched])),
                "arousal": float(np.mean([entry["arousal"] for entry in matched])),
                "dominance": float(np.mean([entry["dominance"] for entry in matched])),
            })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        for dimension in ("valence", "arousal", "dominance"):
            frame[f"cumulative_{dimension}"] = frame[dimension].expanding().mean()
    return frame


def build_va_figure(frame: pd.DataFrame):
    """Render the supplied cumulative valence–arousal circumplex for this session."""
    x = frame["cumulative_valence"].to_numpy()
    y = frame["cumulative_arousal"].to_numpy()
    turns = frame["turn"].to_numpy()
    upper = float(turns.max()) if len(turns) > 1 else float(turns.min() + 1)
    norm = Normalize(vmin=float(turns.min()), vmax=upper)
    figure, axis = plt.subplots(figsize=(7, 4.2))
    for index in range(len(x) - 1):
        axis.plot(
            x[index:index + 2], y[index:index + 2],
            color=CIRCUMPLEX_CMAP(norm(turns[index])), linewidth=3,
            solid_capstyle="round",
        )
    axis.scatter(x[0], y[0], color="white", edgecolors="black", marker="o", s=90,
                 zorder=6, label="Conversation start")
    if len(x) > 1:
        axis.scatter(x[-1], y[-1], color="white", edgecolors="black", marker="*", s=180,
                     zorder=6, label="Latest turn")
    axis.set_xlabel("Valence")
    axis.set_ylabel("Arousal")
    axis.set_title("User valence–arousal trajectory")
    axis.grid(alpha=0.18)
    axis.legend(frameon=True, loc="best")
    scalar = cm.ScalarMappable(cmap=CIRCUMPLEX_CMAP, norm=norm)
    scalar.set_array([])
    figure.colorbar(scalar, ax=axis, label="User turn", shrink=0.82)
    figure.tight_layout()
    return figure


def load_cbt_protocols_text() -> str:
    path = ROOT / "references" / "cbt-protocols.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.exception("Unable to load CBT protocols from %s", path)
        return ""
    protocols = data.get("cbt_protocols", [])

    lines = ["[CBT Protocol Playbook — internal guidance]"]
    for p in protocols:
        name = p.get("name")
        purpose = p.get("purpose")
        if not name:
            continue
        lines.append(f"- {name}: {purpose}")
        for kf in (p.get("key_functions") or [])[:3]:
            lines.append(f"  ◦ {kf}")
        for t in (p.get("techniques") or [])[:3]:
            lines.append(f"  • {t}")
    return "\n".join(lines).strip()


CBT_PLAYBOOK_TEXT = load_cbt_protocols_text()

CASE_STUDIES_PATH = ROOT / "data" / "processed" / "user_case_studies.json"
COREISSUE_CASE = {
    "id": "coreissue_burden",
    "query": "I know I'm a burden to other people, so I keep things to myself and put on a smile, even when I'm not okay.",
    "source": "CoreIssue dataset",
    "core_issue": "Self-esteem and confidence issues",
}
CURATED_CASE_SELECTORS = (
    {"query": "best employee that they have ever had", "source": "realcbt_1_"},
    {"query": "quite busy at school", "source": "realcbt_10_"},
    {"query": "doctors have said that i have cancer again", "source": "realcbt_11_"},
    {"query": "main problem is these voices", "source": "realcbt_12_"},
    {"query": "extra bit of space", "source": "realcbt_14_"},
)


@st.cache_data(show_spinner=False)
def _read_case_studies(path_str: str, modified_ns: int) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """Load and validate case-study cards; modified_ns invalidates Streamlit's cache."""
    del modified_ns
    path = Path(path_str)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.exception("Unable to load case studies from %s", path)
        return [], f"Could not load {path.name}: {exc}"

    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = next(
            (data[key] for key in ("user_case_studies", "case_studies", "data") if isinstance(data.get(key), list)),
            [],
        )
    else:
        rows = []

    available, seen_queries = [], set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        query = str(row.get("user_case_seed_query") or "").strip()
        if not query or query in seen_queries:
            continue
        seen_queries.add(query)
        available.append({
            "id": str(row.get("id") or index),
            "query": query,
            "source": str(row.get("source") or "Unknown").strip(),
            "core_issue": str(row.get("core_issue") or "Unlabeled").strip(),
        })

    cases, missing = [COREISSUE_CASE], []
    for selector in CURATED_CASE_SELECTORS:
        unused = [case for case in available if case not in cases]
        match = next((case for case in unused if selector["query"] in case["query"].casefold()), None)
        if match is None and selector.get("source"):
            match = next(
                (case for case in unused if case["source"].casefold().startswith(selector["source"])),
                None,
            )
        if match is None:
            missing.append(selector)
        else:
            cases.append(match)
    if missing:
        LOGGER.error("Missing curated case studies in %s: %s", path, missing)
        return [], f"Expected six curated examples in {path.name}; found {len(cases)}."
    return cases, None


def load_case_studies(path: Path = CASE_STUDIES_PATH) -> Tuple[List[Dict[str, str]], Optional[str]]:
    if not path.is_file():
        return [], f"Case-study file not found: {path}"
    return _read_case_studies(str(path), path.stat().st_mtime_ns)


THERAPIST_CBT_PROMPT = """
You are a Cognitive Behavior Therapist in a live Cognitive Behavior Therapy (CBT) conversation.
You may receive HIDDEN CONTEXT containing:
- a CBT protocol playbook describing validated intervention strategies
- a structured user schema (triggers, automatic thoughts, emotions, behaviors)
- retrieved clinical concepts relevant to the user’s language
Use all supplied context on every turn.

NON-NEGOTIABLE RULE
Do not mirror, paraphrase, or summarize the patient's message. Interpret meaning and advance
insight; a response that could be mistaken for a reflection is invalid.

INTERNAL CLINICAL REASONING (SILENT)
Before responding:
1. Select the most relevant schema element: trigger, automatic thought, emotion, or behavior.
2. Identify its implicit assumption or cognitive distortion, such as a rule, expectation,
   self-judgment, meaning, or conditional belief.
3. Apply the SINGLE protocol supplied in the hidden context for this turn:
   - V / Validation & Reflection → establish safety before cognitive intervention
   - SQ / Socratic Questioning → examine an assumption through focused inquiry
   - AP / Alternative Perspective → broaden a rigid interpretation with a balanced view
4. Use retrieved concepts only to sharpen interpretation; never name or quote them.

CONTEXT RULES
- Refer to schema elements indirectly, never verbatim.
- Treat retrieved concepts as patterns, never diagnoses.
- Translate technical ideas into lived experience and focus on implications.
- Never name CBT techniques, schemas, diagnoses, or distortions.

RESPONSE
- Write one paragraph of 2–4 sentences with no advice or psychoeducation.
- The response must:
1. Indirectly reference ONE schema element
2. Make an implicit assumption or distortion visible
3. Apply the selected CBT protocol clearly
4. Introduce a NEW interpretation, pattern, or perspective
- Ask exactly one open-ended question only when the protocol requires exploration;
  a question is optional for validation/reflection or cognitive restructuring.
- Reject responses that restate the experience, could apply broadly, or fail to apply a protocol.
""".strip()

PROTOCOL_GUIDANCE = {
    "V": "Validation & Reflection: establish accurate understanding and psychological safety before cognitive intervention.",
    "SQ": "Socratic Questioning: use one focused question to help the user examine an assumption and draw their own conclusion.",
    "AP": "Alternative Perspective: introduce a plausible, balanced interpretation that broadens the user's current view.",
}


CODE_LIKE_RE = re.compile(r"\b\d{4,}\b|[A-Z]{2,}\d{2,}")
FORBIDDEN_PHRASES = ["snomed", "neo4j", "embedding", "rag", "vector", "schema", "playbook", "ontology"]
THERAPIST_GENERATION = {"temperature": 0.15, "num_predict": 140, "top_p": 0.7}
SAFE_REWRITE_PROMPT = (
    "Rewrite as a CBT therapist. One paragraph, 2–4 sentences, no lists, no advice, "
    "no reassurance, exactly one open-ended question, ground the response in the "
    "available user context, and translate clinical language into plain language."
)

def looks_like_therapist_leak(text: str) -> bool:
    t = text.lower()
    if CODE_LIKE_RE.search(text):
        return True
    return any(p in t for p in FORBIDDEN_PHRASES)


def generate_therapist_reply(llm: OllamaChat, messages: List[Dict[str, str]]) -> str:
    reply = llm.chat(messages, **THERAPIST_GENERATION)
    if looks_like_therapist_leak(reply):
        reply = llm.chat(
            messages + [{"role": "system", "content": SAFE_REWRITE_PROMPT}],
            **THERAPIST_GENERATION,
        )
    return reply


def safe_extract_schema(text: str) -> Optional[Dict[str, Any]]:
    """Run the project's gpt-4o-mini cognitive-model extractor when configured."""
    if not USER_SCHEMA_AVAILABLE:
        return None
    try:
        return extract_user_schema(text)
    except Exception:
        LOGGER.exception("OpenAI user-schema extraction failed")
        return None

def sanitize_rag(rag_raw: List[Dict[str, Any]]) -> Dict[str, Any]:
    concepts = []
    for r in rag_raw:
        term = r.get("term")
        if not term:
            continue

        clean_rels = []
        for rel in (r.get("relations") or []):
            tgt = rel.get("target_term")
            if tgt:
                clean_rels.append({
                    "type": rel.get("type"),
                    "target_term": tgt
                })

        concepts.append({
            "term": term,
            "relations": clean_rels
        })

    return {"concepts": concepts[:5]}


@st.cache_resource(show_spinner=False)
def get_neo4j_driver():
    """Reuse Neo4j's thread-safe connection pool across Streamlit reruns."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def retrieve_concepts(text: str, limit: int) -> Optional[Dict[str, Any]]:
    try:
        matches = retrieve_snomed_matches(get_neo4j_driver(), text, k=limit) or []
        return sanitize_rag(matches)
    except Exception:
        LOGGER.exception("SNOMED retrieval failed")
        return None


def build_hidden_context(
    schema: Optional[Dict[str, Any]],
    rag: Optional[Dict[str, Any]],
    use_protocol: bool,
    selected_protocol: Optional[str] = None,
) -> str:
    blocks = []
    if use_protocol and CBT_PLAYBOOK_TEXT:
        blocks.append(CBT_PLAYBOOK_TEXT)
    if use_protocol and selected_protocol in PROTOCOL_GUIDANCE:
        blocks.append(
            "[Selected CBT protocol — required for this turn]\n"
            f"{selected_protocol}: {PROTOCOL_GUIDANCE[selected_protocol]}\n"
            "Apply this protocol in the response without naming it. Do not substitute another protocol."
        )
    if schema:
        blocks.append("[User schema]\n" + json.dumps(schema, ensure_ascii=False))
    if rag and rag.get("concepts"):
        blocks.append("[Retrieved clinical concepts]\n" + json.dumps(rag, ensure_ascii=False))
    return "\n\n".join(blocks)

def audit_grounding(
    therapist_reply: str,
    schema: Optional[Dict[str, Any]],
    rag: Optional[Dict[str, Any]],
) -> Dict[str, bool]:
    """
    - schema_used: any exact schema item substring appears in reply (case-insensitive).
    - rag_used: any first token of a retrieved term appears in reply.
    """
    text = therapist_reply.lower()
    schema_hit = False
    rag_hit = False

    if schema:
        for bucket in ["triggers", "automatic_thoughts", "emotions", "behaviors"]:
            for item in (schema.get(bucket) or []):
                if isinstance(item, str) and item.strip() and item.lower() in text:
                    schema_hit = True
                    break
            if schema_hit:
                break

    if rag and rag.get("concepts"):
        for c in rag["concepts"]:
            term = (c.get("term") or "").lower().strip()
            if term:
                key = term.split()[0]
                if key and key in text:
                    rag_hit = True
                    break

    return {"schema_used": schema_hit, "rag_used": rag_hit}

BRAIN_CIRCUIT_FAVICON_B64 = """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"""
_favicon = Image.open(BytesIO(base64.b64decode(BRAIN_CIRCUIT_FAVICON_B64)))
st.set_page_config(page_title="CBT Lens", page_icon=_favicon, layout="wide", initial_sidebar_state="auto")
CASE_STUDIES, CASE_STUDIES_ERROR = load_case_studies()

CUSTOM_CSS = """
<style>
:root {
  --ink:#211B2E;
  --muted:#756D84;
  --line:#E8E2F2;
  --purple:#7653C6;
  --purple-dark:#6543B3;
  --purple-soft:#F3EEFC;
  --panel:rgba(255,255,255,.82);
}
#MainMenu {visibility:hidden;} footer {visibility:hidden;}
[data-testid="stHeader"] {display:none !important;}
[data-testid="stToolbar"] {display:none !important;}
[data-testid="stApp"] {
  background:
    radial-gradient(circle at 18% 12%, rgba(205,188,241,.32) 0%, transparent 34%),
    radial-gradient(circle at 88% 84%, rgba(191,211,248,.28) 0%, transparent 36%),
    #F8F7FC;
}
[data-testid="stMain"] { background:transparent; }
.block-container {max-width:1040px; padding-top:1.15rem; padding-bottom:5.8rem !important;}

/* Sidebar: quiet, not a giant card */
section[data-testid="stSidebar"] {
  background:rgba(252,251,255,.94);
  border-right:1px solid #ECE7F3;
  box-shadow:5px 0 24px rgba(55,38,91,.04);
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding:1.25rem 1.15rem 2rem;}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] > div {
  background:transparent !important; border:0 !important; box-shadow:none !important; padding:0 !important;
}
section[data-testid="stSidebar"] hr {margin:.9rem 0 !important; border-color:#EEE9F5 !important;}
section[data-testid="stSidebar"], section[data-testid="stSidebar"] * {scrollbar-width:none;}
section[data-testid="stSidebar"]::-webkit-scrollbar, section[data-testid="stSidebar"] *::-webkit-scrollbar {display:none;}
.cbt-brand {margin:.1rem 0 1.35rem;}
.cbt-brand-row {display:flex;align-items:flex-start;gap:11px;}
.cbt-brand-mark {width:38px;height:38px;border-radius:11px;display:flex;align-items:center;justify-content:center;background:linear-gradient(145deg,#7D59C9,#6846B6);color:white;box-shadow:0 6px 16px rgba(108,78,182,.22);flex:0 0 38px;}
.cbt-brand-mark svg {width:21px;height:21px;stroke:#fff;stroke-width:2;fill:none;stroke-linecap:round;stroke-linejoin:round;}
.cbt-brand-copy {min-width:0;}
.cbt-brand-name {font-size:1.02rem;font-weight:760;color:var(--ink);letter-spacing:-.02em;}
.cbt-brand-sub {font-size:.69rem;color:#8A8298;margin-top:1px;}
.cbt-socials {display:flex;gap:8px;margin-top:7px;}
.cbt-socials a {display:inline-flex;align-items:center;justify-content:center;width:21px;height:21px;color:#8F879C;text-decoration:none;border-radius:6px;transition:color .15s ease,background .15s ease;}
.cbt-socials a:hover {color:var(--purple);background:#F3EEFC;}
.cbt-socials svg {width:14px;height:14px;stroke:currentColor;stroke-width:1.8;fill:none;stroke-linecap:round;stroke-linejoin:round;}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p {color:var(--ink);}

/* Keep action-button labels white despite the sidebar's general text rule. */
section[data-testid="stSidebar"] div[data-testid="stButton"] button,
section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] button,
section[data-testid="stSidebar"] div[data-testid="stButton"] button p,
section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] button p,
section[data-testid="stSidebar"] div[data-testid="stButton"] button span,
section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] button span {
  color:#FFFFFF !important;
}

/* Buttons: generic controls */
div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {
  border-radius:11px !important; border:1px solid var(--purple) !important;
  background:var(--purple) !important; color:#fff !important; font-weight:650 !important;
  box-shadow:0 3px 10px rgba(108,78,182,.15); transition:.15s ease !important;
}
div[data-testid="stButton"] button:hover, div[data-testid="stDownloadButton"] button:hover {
  background:var(--purple-dark) !important; border-color:var(--purple-dark) !important;
  box-shadow:0 6px 16px rgba(108,78,182,.20); transform:translateY(-1px);
}

/* Start prompt: intentionally plain, no oversized hero card */
.cbt-start-title {
  margin: clamp(120px, 20vh, 220px) 0 1.15rem;
  color: var(--ink);
  font-size: 1.02rem;
  font-weight: 750;
  letter-spacing: -.018em;
  text-align: center;
}

/* Case cards */
/* beat Streamlit's generic button selector with greater specificity */
[class*="st-key-seedcard_"] div[data-testid="stButton"] button,
[class*="st-key-seedcard_"] button {
  height:224px !important; width:100% !important; padding:20px !important;
  text-align:left !important; align-items:stretch !important; justify-content:flex-start !important;
  background:rgba(255,255,255,.88) !important; color:var(--ink) !important;
  border:1px solid #E5DDF0 !important; border-radius:17px !important;
  box-shadow:0 5px 18px rgba(57,39,91,.055) !important;
  transition:transform .15s ease,box-shadow .15s ease,border-color .15s ease !important;
}
[class*="st-key-seedcard_"] div[data-testid="stButton"] button:hover,
[class*="st-key-seedcard_"] button:hover {
  background:#fff !important; border-color:#BDA7E7 !important;
  box-shadow:0 12px 26px rgba(76,51,118,.11) !important; transform:translateY(-2px);
}
[class*="st-key-seedcard_"] button > div,
[class*="st-key-seedcard_"] button [data-testid="stMarkdownContainer"] {
  display:flex !important;flex-direction:column !important;height:100% !important;width:100% !important;
  gap:10px !important;align-items:flex-start !important;white-space:normal !important;
}
[class*="st-key-seedcard_"] button p {margin:0 !important;width:100%;}
[class*="st-key-seedcard_"] button p:first-child {
  color:#332B40 !important;font-size:.82rem !important;line-height:1.48 !important;font-weight:520 !important;
  display:-webkit-box;-webkit-line-clamp:6;-webkit-box-orient:vertical;overflow:hidden;
}
[class*="st-key-seedcard_"] button p:nth-child(2){margin-top:auto !important;}
[class*="st-key-seedcard_"] button code {
  display:inline-block;padding:4px 9px !important;border-radius:999px;background:#F3EEFC !important;
  border:1px solid #E2D8F4;color:#6846B6 !important;font-size:.60rem !important;letter-spacing:.015em;white-space:normal;
}
[class*="st-key-seedcard_"] button em {font-style:normal;color:#AAA2B4 !important;font-size:.61rem !important;letter-spacing:.02em;}
[class*="st-key-seednav_"] div[data-testid="stButton"] button,
[class*="st-key-seednav_"] button {
  width:36px !important;min-width:36px !important;max-width:36px !important;
  min-height:36px !important;height:36px !important;border-radius:50% !important;padding:0 !important;
  background:rgba(255,255,255,.82) !important;color:#7653C6 !important;border:1px solid #E1D7F1 !important;
  box-shadow:0 3px 10px rgba(76,51,118,.06) !important;
}
[class*="st-key-seednav_"] button:hover {background:rgba(255,255,255,.82) !important;color:#7653C6 !important;border-color:#E1D7F1 !important;box-shadow:0 3px 10px rgba(76,51,118,.06) !important;transform:none !important;}
.seed-dots {text-align:center;color:#D8D0E7;letter-spacing:.28em;font-size:.56rem;margin-top:7px;}
.seed-dots b {color:#7653C6;}
.st-key-seedcontrols {margin:8px auto 0 !important;}
.st-key-seedcontrols [data-testid="stHorizontalBlock"] {
  display:grid !important;grid-template-columns:40px 80px 40px !important;
  justify-content:center !important;align-items:center !important;gap:10px !important;
  width:180px !important;margin:0 auto !important;
}
.st-key-seedcontrols [data-testid="stColumn"] {width:40px !important;min-width:40px !important;}
.st-key-seedcontrols [data-testid="stColumn"]:nth-child(2) {width:80px !important;min-width:80px !important;}
.st-key-seedcontrols .seed-dots {margin-top:0;line-height:36px;}

/* Center the case-study carousel without adding another st.columns nesting level. */
.st-key-seedcarousel {
  width: min(86%, 1160px) !important;
  margin-left: auto !important;
  margin-right: auto !important;
}
.st-key-seedcarousel > div,
.st-key-seedcarousel [data-testid="stVerticalBlock"] {
  width: 100% !important;
}

/* Chat */
[data-testid="stChatInput"] {background:rgba(255,255,255,.90) !important;border:1px solid #D7C7EF !important;border-radius:17px !important;box-shadow:0 6px 18px rgba(65,42,105,.07);position:relative !important;margin-bottom:24px !important;}
[data-testid="stChatInput"]::after {
  content:"This is a research prototype for CBT-grounded LLM evaluation and is not a substitute for professional mental health care.";
  position:absolute;left:0;right:0;top:calc(100% + 7px);text-align:center;color:#8E8799;font-size:.68rem;line-height:1.2;pointer-events:none;
}
.cbt-row {display:flex;margin:10px 0;align-items:flex-end;gap:8px;}
.cbt-row.user {justify-content:flex-end}.cbt-row.assistant{justify-content:flex-start}
.cbt-user-avatar {width:30px;height:30px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;background:var(--purple);box-shadow:0 2px 8px rgba(108,78,182,.20);}
.cbt-bubble {max-width:72%;padding:12px 16px;border-radius:16px;font-size:.92rem;line-height:1.5;}
.cbt-bubble.user {background:#EEE6FA;color:var(--ink);border:1px solid #DFD1F3;border-bottom-right-radius:5px;}
.cbt-bubble.assistant {background:rgba(255,255,255,.88);color:var(--ink);border:1px solid #E8E2F2;border-left:3px solid var(--purple);border-bottom-left-radius:5px;box-shadow:0 3px 12px rgba(66,45,101,.045);}
.cbt-typing {display:inline-flex;gap:4px;padding:4px 2px}.cbt-typing span{width:6px;height:6px;border-radius:50%;background:var(--purple);opacity:.45;animation:cbt-bounce 1.1s infinite ease-in-out}.cbt-typing span:nth-child(2){animation-delay:.15s}.cbt-typing span:nth-child(3){animation-delay:.3s}@keyframes cbt-bounce{0%,80%,100%{transform:translateY(0);opacity:.45}40%{transform:translateY(-4px);opacity:1}}
/* Baseline versus CBT-guided response */
.response-compare {margin:10px 0 12px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;}
.response-panel {background:rgba(255,255,255,.82);border:1px solid #E7E0F1;border-radius:14px;padding:13px 14px;box-shadow:0 3px 12px rgba(66,45,101,.035);}
.response-panel.guided {border-left:3px solid var(--purple);}
.response-panel.baseline {border-left:3px solid #B6AFBF;}
.response-label {font-size:.61rem;letter-spacing:.06em;text-transform:uppercase;font-weight:760;color:#91899D;margin-bottom:7px;}
.response-panel.guided .response-label {color:#7653C6;}
.response-text {font-size:.84rem;line-height:1.52;color:#332B40;}

/* Evaluation */
.eval-wrap {margin:14px 0 6px;padding:14px 15px;background:rgba(255,255,255,.74);border:1px solid #E7E0F1;border-radius:15px;box-shadow:0 4px 16px rgba(66,45,101,.035);}
.eval-title-row {display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:10px;}
.eval-title {font-size:.79rem;font-weight:760;color:var(--ink);}
.eval-subtle {font-size:.62rem;color:#9A92A6;}
.eval-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;}
.turn-summary-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
.eval-card {background:#FBF9FE;border:1px solid #ECE5F5;border-radius:12px;padding:10px 11px;min-height:72px;}
.eval-label {font-size:.59rem;letter-spacing:.045em;text-transform:uppercase;color:#938A9F;font-weight:700;margin-bottom:5px;}
.eval-value {font-size:1.02rem;line-height:1.1;font-weight:780;color:#312842;letter-spacing:-.025em;}
.eval-note {font-size:.59rem;color:#9C94A7;margin-top:5px;line-height:1.3;}
.strategy-options {display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-top:7px;}
.strategy-option {display:inline-block;padding:5px 9px;border-radius:999px;background:#F0E9FB;color:#7653C6;border:1px solid #E0D4F2;font-size:.62rem;font-weight:680;line-height:1.2;}
.strategy-option.selected {background:var(--purple);color:#fff;border-color:var(--purple);}
.concept-groups {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:9px;}
.cognitive-groups {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-top:9px;}
.concept-group {border:1px solid #ECE5F5;border-radius:11px;padding:9px 10px;background:#FBF9FE;min-height:54px;}
.concept-group-title {font-size:.58rem;font-weight:760;text-transform:uppercase;letter-spacing:.045em;margin-bottom:6px;color:#776E82;}
.concept-group.entailment .concept-group-title {color:#39794C;}
.concept-group.contradiction .concept-group-title {color:#A24755;}
.concept-group.neutral .concept-group-title {color:#776E82;}
.concept-item {display:inline-block;font-size:.57rem;line-height:1.3;padding:3px 6px;margin:2px 3px 2px 0;border-radius:999px;background:#F0E9FB;color:#6846B6;border:1px solid #E0D4F2;}
.concept-empty {font-size:.59rem;color:#AAA2B4;}
.session-stat-row {display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;}
.session-stat {background:#F8F5FC;border:1px solid #E9E1F3;border-radius:10px;padding:8px 10px;font-size:.70rem;color:#5F566D;}
.session-stat b {color:#2E263A;}
@media (max-width:900px) {
  .block-container {padding:1rem 1rem 7rem !important;max-width:100%;}
  .st-key-seedcarousel {width:100% !important;}
  [class*="st-key-seedcard_"] button {height:198px !important;padding:16px !important;}
  .eval-grid,.cognitive-groups {grid-template-columns:1fr 1fr;}
  .concept-groups {grid-template-columns:1fr;}
}
@media (max-width:650px) {
  section[data-testid="stSidebar"] {width:min(88vw,320px) !important;min-width:min(88vw,320px) !important;}
  section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding:1rem .9rem 1.4rem;}
  [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {
    position:fixed !important;top:max(12px,env(safe-area-inset-top)) !important;left:12px !important;z-index:1002 !important;
  }
  section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
    position:absolute !important;top:max(12px,env(safe-area-inset-top)) !important;right:12px !important;z-index:3 !important;
  }
  [data-testid="stSidebarCollapsedControl"] button, [data-testid="collapsedControl"] button,
  section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button {
    width:40px !important;height:40px !important;min-width:40px !important;min-height:40px !important;padding:0 !important;
    border:1px solid #E1D7F1 !important;border-radius:12px !important;background:rgba(255,255,255,.96) !important;
    box-shadow:0 5px 16px rgba(66,45,101,.14) !important;color:transparent !important;
  }
  [data-testid="stSidebarCollapsedControl"] button svg, [data-testid="collapsedControl"] button svg,
  section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button svg {display:none !important;}
  [data-testid="stSidebarCollapsedControl"] button::before, [data-testid="collapsedControl"] button::before,
  section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button::before {
    content:"";display:block;width:20px;height:20px;margin:auto;background-repeat:no-repeat;background-position:center;background-size:20px 20px;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%237653C6' stroke-width='2' stroke-linecap='round'%3E%3Cpath d='M4 6h16M4 12h16M4 18h16'/%3E%3C/svg%3E");
  }
  section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button::before {
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%237653C6' stroke-width='2' stroke-linecap='round'%3E%3Cpath d='M18 6 6 18M6 6l12 12'/%3E%3C/svg%3E");
  }
  .cbt-brand {margin-bottom:1rem;}
  .cbt-socials a {width:30px;height:30px;}
  .cbt-socials svg {width:16px;height:16px;}
  .block-container {padding:.75rem .75rem 7.6rem !important;}
  .cbt-start-title {margin:clamp(68px,12vh,105px) .75rem 1rem;font-size:.96rem;line-height:1.35;}
  .st-key-seedcarousel {width:100% !important;padding:0 3px;}
  .st-key-seedcards [data-testid="stHorizontalBlock"] {
    display:flex !important;flex-direction:column !important;flex-wrap:nowrap !important;
    gap:14px !important;overflow:visible !important;padding:2px 1px 7px;
  }
  .st-key-seedcards [data-testid="stColumn"] {
    flex:0 0 auto !important;width:100% !important;min-width:100% !important;
  }
  .st-key-seednav_prev button,.st-key-seednav_next button {
    width:40px !important;max-width:40px !important;height:40px !important;padding:0 !important;border-radius:50% !important;
    background:rgba(255,255,255,.97) !important;box-shadow:0 4px 12px rgba(66,45,101,.10) !important;
  }
  [class*="st-key-seedcard_"] button {height:auto !important;min-height:190px !important;padding:16px !important;}
  [class*="st-key-seedcard_"] button p:first-child {font-size:.80rem !important;-webkit-line-clamp:5;}
  [data-testid="stChatInput"] {border-radius:14px !important;margin-bottom:38px !important;}
  [data-testid="stChatInput"]::after {top:calc(100% + 6px);padding:0 8px;font-size:.59rem;line-height:1.3;}
  .cbt-row {gap:6px;margin:8px 0;}
  .cbt-user-avatar {width:26px;height:26px;}
  .cbt-bubble {max-width:90%;padding:10px 12px;border-radius:13px;font-size:.86rem;line-height:1.48;overflow-wrap:anywhere;}
  .response-compare,.eval-grid,.turn-summary-grid,.cognitive-groups,.concept-groups {grid-template-columns:1fr;}
  .response-panel {padding:12px;}
  .response-text {font-size:.81rem;overflow-wrap:anywhere;}
  .eval-wrap {padding:12px;margin-top:11px;}
  .eval-title-row {align-items:flex-start;flex-direction:column;gap:2px;}
  .eval-card,.concept-group {min-height:0;}
  .session-stat-row {display:grid;grid-template-columns:1fr;width:100%;}
  .session-stat {width:100%;}
  [data-testid="stExpander"] {overflow:hidden;}
  [data-testid="stImage"], [data-testid="stImage"] img {max-width:100% !important;height:auto !important;}
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def escape(text: str) -> str:
    return html_lib.escape(text).replace("\n", "<br>")


USER_AVATAR_SVG = (
    '<div class="cbt-user-avatar">'
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="white" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="8" r="4"></circle>'
    '<path d="M4 20c0-4.4 3.6-8 8-8s8 3.6 8 8"></path>'
    '</svg></div>'
)


def render_bubble(role: str, content: str) -> str:
    if role == "user":
        bubble = f'<div class="cbt-bubble user">{escape(content)}</div>'
        return f'<div class="cbt-row user">{bubble}{USER_AVATAR_SVG}</div>'
    bubble = f'<div class="cbt-bubble assistant">{escape(content)}</div>'
    return f'<div class="cbt-row assistant">{bubble}</div>'


def render_response_comparison(guided_response: str, evaluation: Dict[str, Any]) -> str:
    baseline = evaluation.get("representative_baseline")
    if not baseline:
        return render_bubble("assistant", guided_response)
    return f"""
    <div class="response-compare">
      <div class="response-panel baseline">
        <div class="response-label">Baseline response</div>
        <div class="response-text">{escape(str(baseline))}</div>
      </div>
      <div class="response-panel guided">
        <div class="response-label">CBT-guided framework response</div>
        <div class="response-text">{escape(guided_response)}</div>
      </div>
    </div>
    """


COGNITIVE_MODEL_FIELDS = (
    ("triggers", "Triggers"),
    ("automatic_thoughts", "Automatic thoughts"),
    ("emotions", "Emotions"),
    ("behaviors", "Behaviors"),
)
def render_turn_evaluation(
    evaluation: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
) -> str:
    force = evaluation.get("force_pct")
    selected = evaluation.get("selected_protocol") if evaluation.get("protocol_applied") else None
    strategy_options = "".join(
        f'<span class="strategy-option {"selected" if strategy == selected else ""}">{strategy}</span>'
        for strategy in ("V", "SQ", "AP")
    )
    force_text = f"{force:.3f}%" if isinstance(force, (int, float)) else "—"
    concept_groups = evaluation.get("snomed_groups") or {}
    group_html = "".join(
        f'<div class="concept-group {relation}">'
        f'<div class="concept-group-title">{label}</div>'
        + (
            "".join(f'<span class="concept-item">{html_lib.escape(term)}</span>' for term in concept_groups.get(relation, []))
            or '<span class="concept-empty">None</span>'
        )
        + "</div>"
        for relation, label in (
            ("entailment", "Entailment"),
            ("contradiction", "Contradiction"),
            ("neutral", "Neutral"),
        )
    )
    cognitive_html = "".join(
        f'<div class="concept-group">'
        f'<div class="concept-group-title">{label}</div>'
        + (
            "".join(
                f'<span class="concept-item">{html_lib.escape(item)}</span>'
                for item in (schema or {}).get(bucket, [])
                if isinstance(item, str) and item.strip()
            )
            or '<span class="concept-empty">None identified</span>'
        )
        + "</div>"
        for bucket, label in COGNITIVE_MODEL_FIELDS
    )
    return f"""
    <div class="eval-wrap">
      <div class="eval-title-row">
        <div class="eval-title">Turn evaluation</div>
        <div class="eval-subtle">CBT Lens research instrumentation</div>
      </div>
      <div class="eval-grid turn-summary-grid">
        <div class="eval-card">
          <div class="eval-label">CBT strategy selected</div>
          <div class="strategy-options">{strategy_options}</div>
        </div>
        <div class="eval-card">
          <div class="eval-label">Protocol Leverage Force (F)</div>
          <div class="eval-value">{force_text}</div>
          <div class="eval-note">Behavioral reorientation</div>
        </div>
      </div>
      <div class="eval-label" style="margin-top:11px">Cognitive model</div>
      <div class="cognitive-groups">{cognitive_html}</div>
      <div class="eval-label" style="margin-top:11px">Retrieved SNOMED concepts</div>
      <div class="concept-groups">{group_html}</div>
    </div>
    """


TYPING_HTML = """
<div class="cbt-row assistant">
  <div class="cbt-bubble assistant">
    <div class="cbt-typing"><span></span><span></span><span></span></div>
  </div>
</div>
"""

st.session_state.setdefault("messages", [])
st.session_state.setdefault("therapist_chat", [{"role": "system", "content": THERAPIST_CBT_PROMPT}])
st.session_state.setdefault("cbt_context", [])
st.session_state.setdefault("case_page", 0)
try:
    session_nclid_stats = compute_nclid(st.session_state.cbt_context)
except Exception:
    LOGGER.exception("Unable to compute session nCLiD")
    session_nclid_stats = None

MARKDOWN_ESCAPE_RE = re.compile(r"([\\`*_{}\[\]()#+\-.!|>])")


def _escape_markdown(value: str) -> str:
    return MARKDOWN_ESCAPE_RE.sub(r"\\\1", value)


def _case_label(case: Dict[str, str]) -> str:
    core_issue = case["core_issue"].replace("`", "'")
    return (
        f"“{_escape_markdown(case['query'])}”\n\n"
        f"`{core_issue}`\n\n"
        f"*Source: {_escape_markdown(case['source'])}*"
    )


def render_case_cards() -> None:
    if not CASE_STUDIES:
        st.error(CASE_STUDIES_ERROR or "Case studies are unavailable.")
        return

    total_pages = max(1, (len(CASE_STUDIES) + CARDS_PER_PAGE - 1) // CARDS_PER_PAGE)
    page = st.session_state.case_page % total_pages

    with st.container(key="seedcarousel"):
        with st.container(key="seedcards"):
            start = page * CARDS_PER_PAGE
            visible = CASE_STUDIES[start:start + CARDS_PER_PAGE]
            cols = st.columns(CARDS_PER_PAGE, gap="medium")
            for offset, (col, case) in enumerate(zip(cols, visible)):
                with col:
                    if st.button(_case_label(case), key=f"seedcard_{start + offset}", use_container_width=True):
                        st.session_state.pending_case = case
                        st.rerun()

        with st.container(key="seedcontrols"):
            nav_l, dots_col, nav_r = st.columns([1, 2, 1], vertical_alignment="center")
            with nav_l:
                if st.button("‹", key="seednav_prev"):
                    st.session_state.case_page = (page - 1) % total_pages
                    st.rerun()
            with dots_col:
                dots = " ".join("<b>●</b>" if i == page else "●" for i in range(total_pages))
                st.markdown(f'<div class="seed-dots">{dots}</div>', unsafe_allow_html=True)
            with nav_r:
                if st.button("›", key="seednav_next"):
                    st.session_state.case_page = (page + 1) % total_pages
                    st.rerun()


with st.sidebar:
    st.markdown(
        """
        <div class="cbt-brand">
          <div class="cbt-brand-row">
            <div class="cbt-brand-mark" aria-hidden="true">
              <svg viewBox="0 0 24 24" aria-label="Brain circuit">
                <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/>
                <path d="M9 13a4.5 4.5 0 0 0 3-4"/>
                <path d="M6.003 5.125A3 3 0 0 0 6.401 6.5"/>
                <path d="M3.477 10.896a4 4 0 0 1 .585-.396"/>
                <path d="M6 18a4 4 0 0 1-1.967-.516"/>
                <path d="M12 13h4"/>
                <path d="M12 18h6a2 2 0 0 1 2 2v1"/>
                <path d="M12 8h8"/>
                <path d="M16 8V5a2 2 0 0 1 2-2"/>
                <circle cx="16" cy="13" r=".5" fill="white" stroke="none"/>
                <circle cx="18" cy="3" r=".5" fill="white" stroke="none"/>
                <circle cx="20" cy="21" r=".5" fill="white" stroke="none"/>
                <circle cx="20" cy="8" r=".5" fill="white" stroke="none"/>
              </svg>
            </div>
            <div class="cbt-brand-copy">
              <div class="cbt-brand-name">CBT Lens</div>
              <div class="cbt-brand-sub">CBT-guided affective reasoning</div>
              <div class="cbt-socials">
                <a href="https://github.com/cbt-llm/cbt-llm" target="_blank" rel="noopener noreferrer" aria-label="CBT LLM on GitHub" title="GitHub">
                  <svg viewBox="0 0 24 24"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3.3-.4 6.8-1.6 6.8-7A5.4 5.4 0 0 0 19.4 4 5 5 0 0 0 19.3.5S18.2.1 15 1.8a13.4 13.4 0 0 0-7 0C4.8.1 3.7.5 3.7.5A5 5 0 0 0 3.6 4a5.4 5.4 0 0 0-1.4 3.7c0 5.4 3.5 6.6 6.8 7A4.8 4.8 0 0 0 8 18v4"/><path d="M8 19c-3 .9-3-1.5-4-2"/></svg>
                </a>
                <a href="https://www.linkedin.com/in/vaishnavi-sinha" target="_blank" rel="noopener noreferrer" aria-label="Vaishnavi Sinha on LinkedIn" title="LinkedIn">
                  <svg viewBox="0 0 24 24"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-4 0v7h-4v-7a6 6 0 0 1 6-6z"/><rect width="4" height="12" x="2" y="9"/><circle cx="4" cy="4" r="2"/></svg>
                </a>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=False):
        st.markdown("**Session Controls**")
        therapist_model = st.selectbox(
            "Select model",
            options=MODEL_OPTIONS,
            index=0,
        )
        st.divider()
        st.markdown("**Select components**")
        use_schema = st.toggle(
            "Cognitive Model",
            value=USER_SCHEMA_AVAILABLE,
            disabled=not USER_SCHEMA_AVAILABLE,
            help="Requires OPENAI_API_KEY and uses the project's gpt-4o-mini schema extractor.",
        )
        if not USER_SCHEMA_AVAILABLE:
            st.caption("Cognitive Model requires `OPENAI_API_KEY`.")
        use_rag = st.toggle("SNOMED Concepts", value=True)
        use_protocol = st.toggle("CBT Protocols", value=True)
        k = st.slider("Retrieved concepts", min_value=1, max_value=5, value=5, step=1, disabled=not use_rag)
        st.divider()
        exchanges = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        st.markdown("**Chat session transcript**")
        st.caption(f"{exchanges} exchange{'s' if exchanges != 1 else ''} recorded")
        export = {
            "app": "CBT Lens",
            "therapist_model": therapist_model,
            "therapist_mode": "cbt",
            "turns_so_far": exchanges,
            "transcript": st.session_state.messages,
            "cbt_context": st.session_state.cbt_context,
            "session_nclid": session_nclid_stats,
        }
        st.download_button("↓  Export JSON", data=json.dumps(export, ensure_ascii=False, indent=2), file_name="cbt_lens_session.json", mime="application/json", use_container_width=True)
        if st.button("↻  New conversation", use_container_width=True):
            st.session_state.clear()
            st.rerun()


pending_case = st.session_state.get("pending_case")
if not st.session_state.messages and not pending_case:
    st.markdown('<div class="cbt-start-title">Start with a thought or select from an existing case study</div>', unsafe_allow_html=True)
    render_case_cards()
elif st.session_state.messages:
    context_index = 0
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(render_bubble("user", msg["content"]), unsafe_allow_html=True)
            continue
        context = st.session_state.cbt_context[context_index] if context_index < len(st.session_state.cbt_context) else {}
        evaluation = context.get("evaluation") or {}
        st.markdown(render_response_comparison(msg["content"], evaluation), unsafe_allow_html=True)
        if evaluation and not evaluation.get("error"):
            st.markdown(
                render_turn_evaluation(
                    evaluation,
                    context.get("schema"),
                ),
                unsafe_allow_html=True,
            )
        context_index += 1

chat_input = st.chat_input("How are you feeling today?")
user_input = pending_case["query"] if pending_case else chat_input

if user_input and not therapist_model:
    st.warning("Select a model in Session Controls to begin.")
    st.session_state.pop("pending_case", None)
    user_input = None

if user_input:
    selected_case = st.session_state.pop("pending_case", None)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(render_bubble("user", user_input), unsafe_allow_html=True)
    visible_history_for_baseline = list(st.session_state.messages)

    typing_slot = st.empty()
    typing_slot.markdown(TYPING_HTML, unsafe_allow_html=True)

    llm = OllamaChat(therapist_model)

    schema = safe_extract_schema(user_input) if use_schema else None

    rag_safe = retrieve_concepts(user_input, k) if use_rag else None

    try:
        protocol_eval = select_protocol_and_classify_concepts(
            llm,
            user_input,
            (rag_safe or {}).get("concepts", []),
            schema,
            visible_history_for_baseline,
        )
    except (requests.RequestException, RuntimeError):
        LOGGER.exception("Protocol selection failed")
        typing_slot.empty()
        st.error("The CBT protocol could not be selected. Please try this turn again.")
        st.session_state.messages.pop()
        st.stop()

    selected_protocol = protocol_eval["selected_protocol"] if use_protocol else None
    hidden = build_hidden_context(schema, rag_safe, use_protocol, selected_protocol)

    therapist_input = list(st.session_state.therapist_chat)
    if hidden:
        therapist_input.append({"role": "system", "content": hidden})
    therapist_input.append({"role": "user", "content": user_input})

    try:
        therapist_reply = generate_therapist_reply(llm, therapist_input)
    except (requests.RequestException, RuntimeError):
        LOGGER.exception("Therapist response generation failed")
        typing_slot.empty()
        st.error("The selected model is unavailable. Confirm that Ollama is running and try again.")
        st.stop()

    typing_slot.markdown(render_bubble("assistant", therapist_reply), unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": therapist_reply})

    st.session_state.therapist_chat.extend([
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": therapist_reply},
    ])

    audit = audit_grounding(therapist_reply, schema, rag_safe)

    evaluation: Dict[str, Any] = {
        **protocol_eval,
        "selected_protocol": selected_protocol,
        "protocol_applied": use_protocol,
    }
    with st.spinner("Computing research evaluation…"):
        try:
            baseline_replies = generate_baseline_replies(llm, visible_history_for_baseline, user_input)
            evaluation["baseline_replies"] = baseline_replies
            evaluation["representative_baseline"] = baseline_replies[0] if baseline_replies else None
        except Exception:
            LOGGER.exception("Baseline generation failed")
            evaluation["baseline_replies"] = []
            evaluation["representative_baseline"] = None
            evaluation["baseline_error"] = "Baseline response unavailable for this turn."

    new_context = {
        "patient_text": user_input,
        "case_study": selected_case,
        "schema": schema,
        "rag": rag_safe,
        "therapist_reply": therapist_reply,
        "grounding_audit": audit,
        "evaluation": evaluation,
    }
    st.session_state.cbt_context.append(new_context)
    try:
        for context, force in zip(
            st.session_state.cbt_context,
            compute_force_series(st.session_state.cbt_context),
        ):
            context.setdefault("evaluation", {})["force_pct"] = force
    except Exception:
        LOGGER.exception("Unable to compute Protocol Leverage Force")
        evaluation["force_pct"] = None
    st.rerun()

if st.session_state.cbt_context:
    valid_evals = [
        c.get("evaluation") for c in st.session_state.cbt_context
        if c.get("evaluation") and not c.get("evaluation", {}).get("error")
    ]
    with st.expander("Session evaluation", expanded=False):
        force_vals = [
            evaluation.get("force_pct") for evaluation in valid_evals
            if isinstance(evaluation.get("force_pct"), (int, float))
        ]
        mean_force = sum(force_vals) / len(force_vals) if force_vals else None
        force_text = f"{mean_force:.3f}%" if isinstance(mean_force, (int, float)) else "Pending"
        session_nclid = (session_nclid_stats or {}).get("nCLiD")
        nclid_text = f"{session_nclid:.3f}" if isinstance(session_nclid, (int, float)) else "Waiting for second turn"
        st.markdown(
            f'<div class="session-stat-row">'
            f'<div class="session-stat"><b>Mean Protocol Leverage Force (F)</b> &nbsp; {force_text}</div>'
            f'<div class="session-stat"><b>nCLiD</b> &nbsp; {nclid_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Affective state · Valence–arousal trajectory**")
        try:
            vad = load_nrc_vad(str(_nrc_vad_directory()))
            sentiment = extract_user_turn_sentiments(st.session_state.cbt_context, vad)
            if sentiment.empty:
                st.caption("No session tokens matched the NRC VAD Lexicon yet.")
            else:
                figure = build_va_figure(sentiment)
                st.pyplot(figure, use_container_width=True)
                plt.close(figure)
        except (OSError, ValueError, pd.errors.ParserError):
            LOGGER.exception("Unable to render NRC VAD trajectory")
            st.warning("NRC VAD Lexicon files are unavailable. Set NRC_VAD_DIR to the lexicon directory.")
        if session_nclid is not None:
            st.caption(f"Lower nCLiD indicates greater linguistic entrainment · sentence-BERT: {SBERT_MODEL}.")

