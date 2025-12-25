"""

User interface for live CBT conversations.

Run:
streamlit run cbt-llm/app.py
"""
import os
import json
import re
from typing import Any, Dict, List, Optional

import streamlit as st
import requests
from neo4j import GraphDatabase

from cbt_llm.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, ROOT
from cbt_llm.retrieve_snomed import retrieve_snomed_matches

try:
    from cbt_llm.user_schema import extract_user_schema
except Exception:
    extract_user_schema = None


class OllamaChat:
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
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
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        return (r.json().get("message") or {}).get("content", "").strip()

def load_cbt_protocols_text() -> str:
    path = ROOT / "references" / "cbt-protocols.json"
    if not path.exists():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
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


THERAPIST_CBT_PROMPT = """
You are a Cognitive Behavior Therapist in a live Cognitive Behavior Therapy (CBT) conversation.

You are given HIDDEN CONTEXT that includes:
- a CBT protocol playbook describing validated intervention strategies
- a structured user schema (triggers, automatic thoughts, emotions, behaviors)
- retrieved clinical concepts relevant to the user’s language

You MUST use this context on EVERY turn.

━━━━━━━━━━━━━━━━━━━
NON-NEGOTIABLE RULE
━━━━━━━━━━━━━━━━━━━

You MUST NOT mirror, paraphrase, or summarize the patient’s message.

If your response could be mistaken for a reflection of what the patient just said,
the response is INVALID.

Your task is to INTERPRET meaning and advance insight,
not to echo content.

━━━━━━━━━━━━━━━━━━━
INTERNAL CLINICAL REASONING (SILENT)
━━━━━━━━━━━━━━━━━━━

Before writing your response, do ALL of the following internally:

1. Select the MOST RELEVANT schema element
   (trigger OR automatic thought OR emotion OR behavior)

2. Identify the implicit assumption or cognitive distortion beneath it
   (e.g., rules, expectations, self-judgments, meanings, conditional beliefs)

3. Select the SINGLE most appropriate CBT protocol:
   - validate_and_reflect → when emotional safety or alignment is primary
   - socratic_questioning → when assumptions need to be examined
   - cognitive_reframing → when interpretations are rigid or limiting

4. Use the most relevant retrieved concepts ONLY to sharpen interpretation
   (they are support signals; do NOT name or quote them)

━━━━━━━━━━━━━━━━━━━
HOW TO USE CONTEXT
━━━━━━━━━━━━━━━━━━━

- Refer to schema elements INDIRECTLY, never verbatim
- Treat retrieved concepts as patterns, never diagnoses
- Translate technical ideas into lived experience
- Focus on what the experience IMPLIES, not what it IS
- Never name CBT techniques, schemas, diagnoses, or distortions

━━━━━━━━━━━━━━━━━━━
RESPONSE CONSTRAINTS
━━━━━━━━━━━━━━━━━━━

- ONE paragraph
- 2–4 sentences total
- NO advice
- NO psychoeducation

━━━━━━━━━━━━━━━━━━━
CONTENT REQUIREMENTS
━━━━━━━━━━━━━━━━━━━

Your response MUST:
1. Indirectly reference ONE schema element
2. Make an implicit assumption or distortion visible
3. Apply the selected CBT protocol clearly
4. Introduce a NEW interpretation, pattern, or perspective

Question use:
- Ask EXACTLY ONE open-ended question ONLY if the chosen protocol requires exploration
  (e.g., socratic_questioning).
- If using validate_and_reflect or cognitive_reframing, a question is OPTIONAL.

If the response restates the patient’s experience, it is wrong.
If the response could apply to many people, it is wrong.
If the CBT protocol is not clearly applied, it is wrong.
""".strip()


CODE_LIKE_RE = re.compile(r"\b\d{4,}\b|[A-Z]{2,}\d{2,}")
FORBIDDEN_PHRASES = ["snomed", "neo4j", "embedding", "rag", "vector", "schema", "playbook", "ontology"]

def looks_like_therapist_leak(text: str) -> bool:
    t = text.lower()
    if CODE_LIKE_RE.search(text):
        return True
    return any(p in t for p in FORBIDDEN_PHRASES)


def safe_extract_schema(text: str) -> Optional[Dict[str, Any]]:
    if extract_user_schema is None:
        return None
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        return extract_user_schema(text)
    except Exception:
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

    return {"concepts": concepts[:10]}


def build_hidden_context(
    schema: Optional[Dict[str, Any]],
    rag: Optional[Dict[str, Any]],
    use_protocol: bool,
) -> str:
    blocks = []
    if use_protocol and CBT_PLAYBOOK_TEXT:
        blocks.append(CBT_PLAYBOOK_TEXT)
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

st.set_page_config(page_title="CBT LLM", layout="wide")
st.title("CBT LLM")

with st.sidebar:
    st.header("Controls")

    therapist_model = st.text_input("Therapist model (Ollama)", value="gemma2:9b")
    use_schema = st.checkbox("User schema", value=True)
    use_rag = st.checkbox("Use SNOMED retrieval", value=True)
    use_protocol = st.checkbox("Use CBT Strategy", value=True)

    k = st.slider("Top-k concepts", min_value=1, max_value=15, value=5, step=1)

    st.divider()
    st.subheader("Latest debug")
    debug_slot = st.empty()

    st.divider()
    if st.button("Reset chat"):
        st.session_state.clear()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []
if "therapist_chat" not in st.session_state:
    st.session_state.therapist_chat: List[Dict[str, str]] = [
        {"role": "system", "content": THERAPIST_CBT_PROMPT}
    ]
if "cbt_context" not in st.session_state:
    st.session_state.cbt_context: List[Dict[str, Any]] = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("How are you feeling today?")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    schema = safe_extract_schema(user_input) if use_schema else None

    rag_safe = None
    rag_raw = None
    if use_rag:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        try:
            rag_raw = retrieve_snomed_matches(driver, user_input, k=k) or []
        finally:
            driver.close()
        rag_safe = sanitize_rag(rag_raw)

    hidden = build_hidden_context(schema, rag_safe, use_protocol)

    print("\n" + "=" * 90)
    print("[NEW TURN] user_input:", user_input)
    print("[SCHEMA]", json.dumps(schema, ensure_ascii=False, indent=2) if schema else None)
    print("[RAG]", json.dumps(rag_safe, ensure_ascii=False, indent=2) if rag_safe else None)
    print("=" * 90 + "\n")

    debug_payload = {
        "schema": schema,
        "retrieved_terms": [c["term"] for c in (rag_safe or {}).get("concepts", [])]
    }
    debug_slot.json(debug_payload)


    llm = OllamaChat(therapist_model)

    therapist_input = list(st.session_state.therapist_chat)
    if hidden:
        therapist_input.append({"role": "system", "content": hidden})
    therapist_input.append({"role": "user", "content": user_input})

    therapist_reply = llm.chat(
        therapist_input,
        temperature=0.15,
        num_predict=140,
        top_p=0.7,
    )

    if looks_like_therapist_leak(therapist_reply):
        therapist_reply = llm.chat(
            therapist_input + [{
                "role": "system",
                "content": (
                    "Rewrite as a CBT therapist. One paragraph, 2–4 sentences, "
                    "no lists, no advice, no reassurance, exactly one open-ended question, "
                    "must ground in at least one schema item, translate any clinical term into plain language."
                )
            }],
            temperature=0.15,
            num_predict=140,
            top_p=0.7,
        )

    st.session_state.messages.append({"role": "assistant", "content": therapist_reply})
    with st.chat_message("assistant"):
        st.markdown(therapist_reply)

    st.session_state.therapist_chat.append({"role": "user", "content": user_input})
    st.session_state.therapist_chat.append({"role": "assistant", "content": therapist_reply})

    audit = audit_grounding(therapist_reply, schema, rag_safe)
    st.session_state.cbt_context.append({
        "patient_text": user_input,
        "schema": schema,
        "rag": rag_safe,
        "therapist_reply": therapist_reply,
        "grounding_audit": audit,
    })

    with st.sidebar:
        st.caption(f"Grounding audit: schema_used={audit['schema_used']} | rag_used={audit['rag_used']}")

with st.sidebar:
    st.divider()
    st.subheader("Export")
    export = {
        "therapist_model": therapist_model,
        "therapist_mode": "cbt",
        "turns_so_far": len([m for m in st.session_state.messages if m["role"] == "assistant"]),
        "transcript": st.session_state.messages,
        "cbt_context": st.session_state.cbt_context,
    }
    st.download_button(
        "Download JSON",
        data=json.dumps(export, ensure_ascii=False, indent=2),
        file_name="cbt_chat_export.json",
        mime="application/json",
    )
