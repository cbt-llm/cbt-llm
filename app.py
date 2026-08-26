"""

User interface for live CBT conversations.

Run:
streamlit run cbt-llm/app.py
"""
import os
import json
import re
import html as html_lib
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
   - cognitive_restructuring → when interpretations are rigid or limiting

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
- If using validate_and_reflect or cognitive_restructuring, a question is OPTIONAL.

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

st.set_page_config(page_title="CBT LLM", page_icon="🪷", layout="wide")

CUSTOM_CSS = """
<style>
:root {
  --cbt-card: #FFFFFF;
  --cbt-border: #E5DFF2;
  --cbt-text: #241F33;
  --cbt-text-soft: #6B6480;
  --cbt-primary: #6C4EB6;
  --cbt-primary-light: #EFE7FB;
  --cbt-accent: #9B7FD4;
  --cbt-user-bg: rgba(155, 127, 212, 0.16);
  --cbt-assistant-bg: rgba(255, 255, 255, 0.66);
  --cbt-success: #2F855A;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.block-container { padding-top: 1.4rem; max-width: 920px; }

[data-testid="stApp"] {
  background:
    radial-gradient(circle at 12% 8%, rgba(199, 174, 240, 0.35) 0%, transparent 45%),
    radial-gradient(circle at 88% 92%, rgba(163, 196, 243, 0.30) 0%, transparent 45%),
    #F7F5FC;
}

[data-testid="stMain"] {
  background: rgba(255, 255, 255, 0.60);
}

section[data-testid="stSidebar"] {
  background: rgba(255, 255, 255, 0.82);
  backdrop-filter: blur(18px) saturate(160%);
  border-right: 1px solid rgba(108,78,182,0.28);
  box-shadow: 6px 0 28px rgba(108,78,182,0.10);
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
  padding-top: 0.75rem;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
  border-color: rgba(108,78,182,0.35) !important;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] > div {
  background: rgba(255, 255, 255, 0.55);
  backdrop-filter: blur(10px);
  box-shadow: 0 4px 18px rgba(108,78,182,0.08);
}

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] * {
  scrollbar-width: none;
  -ms-overflow-style: none;
}
section[data-testid="stSidebar"]::-webkit-scrollbar,
section[data-testid="stSidebar"] *::-webkit-scrollbar {
  display: none;
  width: 0;
  height: 0;
}

[data-testid="stChatInput"] {
  background: rgba(255, 255, 255, 0.65) !important;
  backdrop-filter: blur(14px);
  border: 1px solid rgba(108,78,182,0.22) !important;
  border-radius: 18px !important;
  box-shadow: 0 4px 16px rgba(108,78,182,0.08);
}

div[data-testid="stButton"] button,
div[data-testid="stDownloadButton"] button {
  border-radius: 10px !important;
  border: 1px solid var(--cbt-primary) !important;
  background: var(--cbt-primary) !important;
  color: #FFFFFF !important;
  font-weight: 600 !important;
  transition: all 0.18s ease !important;
  box-shadow: 0 2px 10px rgba(108,78,182,0.20);
}
div[data-testid="stButton"] button:hover,
div[data-testid="stDownloadButton"] button:hover {
  background: rgba(108,78,182,0.82) !important;
  border-color: rgba(108,78,182,0.9) !important;
  box-shadow: 0 6px 18px rgba(108,78,182,0.28) !important;
  transform: translateY(-1px);
}
div[data-testid="stButton"] button:active,
div[data-testid="stDownloadButton"] button:active {
  transform: translateY(0);
  background: #5A3D9E !important;
  box-shadow: 0 2px 6px rgba(108,78,182,0.22) !important;
}
div[data-testid="stButton"] button:focus:not(:active),
div[data-testid="stDownloadButton"] button:focus:not(:active) {
  border-color: rgba(108,78,182,0.9) !important;
  box-shadow: 0 0 0 3px rgba(108,78,182,0.20) !important;
}

.cbt-welcome {
  background: rgba(255, 255, 255, 0.55);
  backdrop-filter: blur(16px) saturate(160%);
  border: 1px solid rgba(108,78,182,0.18);
  border-radius: 18px;
  padding: 20px 28px;
  text-align: center;
  margin-bottom: 10px;
  box-shadow: 0 6px 20px rgba(108,78,182,0.08);
}
.cbt-welcome-icon {
  width: 38px; height: 38px; border-radius: 50%;
  margin: 0 auto 10px auto;
  display: flex; align-items: center; justify-content: center; gap: 5px;
  background: linear-gradient(135deg, rgba(108,78,182,0.16), rgba(155,127,212,0.08));
  border: 1px solid rgba(108,78,182,0.20);
  box-shadow: 0 4px 12px rgba(108,78,182,0.12), inset 0 1px 0 rgba(255,255,255,0.7);
}
.cbt-welcome-icon span {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--cbt-primary); opacity: 0.7;
}
.cbt-welcome-icon span:nth-child(2) { opacity: 1; transform: scale(1.2); }
.cbt-welcome h3 { margin: 2px 0 4px 0; color: var(--cbt-text); font-weight: 650; font-size: 1.05rem; letter-spacing: -0.01em; }
.cbt-welcome p { color: var(--cbt-text-soft); max-width: 460px; margin: 0 auto; font-size: 0.85rem; line-height: 1.45; }

.cbt-row { display: flex; margin: 10px 0; align-items: flex-end; gap: 8px; }
.cbt-row.user { justify-content: flex-end; }
.cbt-row.assistant { justify-content: flex-start; }
.cbt-user-avatar {
  width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  background: var(--cbt-primary);
  box-shadow: 0 2px 8px rgba(108,78,182,0.25);
}
.cbt-bubble {
  max-width: 70%;
  padding: 12px 16px;
  border-radius: 16px;
  font-size: 0.95rem;
  line-height: 1.5;
  backdrop-filter: blur(10px);
}
.cbt-bubble.user {
  background: var(--cbt-user-bg);
  color: var(--cbt-text);
  border: 1px solid rgba(108,78,182,0.16);
  border-bottom-right-radius: 4px;
}
.cbt-bubble.assistant {
  background: var(--cbt-assistant-bg);
  color: var(--cbt-text);
  border: 1px solid var(--cbt-border);
  border-left: 3px solid var(--cbt-primary);
  border-bottom-left-radius: 4px;
  box-shadow: 0 2px 10px rgba(108,78,182,0.06);
}

.cbt-typing { display: inline-flex; gap: 4px; padding: 4px 2px; }
.cbt-typing span {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--cbt-primary); opacity: 0.5;
  animation: cbt-bounce 1.1s infinite ease-in-out;
}
.cbt-typing span:nth-child(2) { animation-delay: 0.15s; }
.cbt-typing span:nth-child(3) { animation-delay: 0.3s; }
@keyframes cbt-bounce {
  0%, 80%, 100% { transform: translateY(0); opacity: 0.5; }
  40% { transform: translateY(-4px); opacity: 1; }
}

.cbt-chip {
  display: inline-block; background: var(--cbt-primary-light); color: var(--cbt-primary);
  border-radius: 999px; padding: 3px 10px; font-size: 0.78rem; margin: 2px 4px 2px 0;
  border: 1px solid rgba(108,78,182,0.15);
}
.cbt-badge { display: inline-flex; align-items: center; gap: 5px; font-size: 0.8rem; font-weight: 600; padding: 3px 10px; border-radius: 999px; margin-right: 8px; }
.cbt-badge.on { background: #E6F6EC; color: var(--cbt-success); }
.cbt-badge.off { background: #F1F3F5; color: #8A97A3; }

.cbt-footer-note { text-align: center; color: var(--cbt-text-soft); font-size: 0.78rem; margin-top: 18px; padding-bottom: 8px; }
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


TYPING_HTML = """
<div class="cbt-row assistant">
  <div class="cbt-bubble assistant">
    <div class="cbt-typing"><span></span><span></span><span></span></div>
  </div>
</div>
"""

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []
if "therapist_chat" not in st.session_state:
    st.session_state.therapist_chat: List[Dict[str, str]] = [
        {"role": "system", "content": THERAPIST_CBT_PROMPT}
    ]
if "cbt_context" not in st.session_state:
    st.session_state.cbt_context: List[Dict[str, Any]] = []

with st.sidebar:
    st.markdown("### CBT LLM")
    st.caption("Presenter controls")

    MODEL_OPTIONS = ["gemma2:9b", "gemma3:12b", "mistral:7b", "deepseek-r1:8b", "gpt-oss:20b"]

    with st.container(border=True):
        st.markdown("**Session**")
        therapist_model = st.selectbox("AI model (Ollama)", options=MODEL_OPTIONS, index=0)
        if st.button("Reset conversation", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        st.divider()

        st.markdown("**Clinical intelligence**")
        use_schema = st.toggle("User schema modeling", value=True)
        use_rag = st.toggle("SNOMED concept retrieval", value=True)
        use_protocol = st.toggle("CBT protocol grounding", value=True)
        k = st.slider(
            "Top-k concepts", min_value=1, max_value=15, value=5, step=1, disabled=not use_rag
        )

        st.divider()

        st.markdown("**📤 Export**")
        st.caption(
            f"{len([m for m in st.session_state.messages if m['role'] == 'assistant'])} exchange(s) recorded"
        )
        export = {
            "therapist_model": therapist_model,
            "therapist_mode": "cbt",
            "turns_so_far": len([m for m in st.session_state.messages if m["role"] == "assistant"]),
            "transcript": st.session_state.messages,
            "cbt_context": st.session_state.cbt_context,
        }
        st.download_button(
            "Download transcript (JSON)",
            data=json.dumps(export, ensure_ascii=False, indent=2),
            file_name="cbt_chat_export.json",
            mime="application/json",
            use_container_width=True,
        )

user_input = st.chat_input("Share what's on your mind…")

if not st.session_state.messages and not user_input:
    st.markdown(
        """
    <div class="cbt-welcome">
      <div class="cbt-welcome-icon"><span></span><span></span><span></span></div>
      <h3>Start a conversation</h3>
      <p>Share what's on your mind — I'll listen, reflect, and gently explore the
      thoughts and feelings behind it.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
        st.markdown(render_bubble(msg["role"], msg["content"]), unsafe_allow_html=True)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(render_bubble("user", user_input), unsafe_allow_html=True)

    typing_slot = st.empty()
    typing_slot.markdown(TYPING_HTML, unsafe_allow_html=True)

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

    typing_slot.markdown(render_bubble("assistant", therapist_reply), unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": therapist_reply})

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

if st.session_state.cbt_context:
    latest = st.session_state.cbt_context[-1]
    schema = latest.get("schema")
    rag = latest.get("rag")
    audit = latest.get("grounding_audit") or {}

    with st.expander("🔬 Behind the scenes — latest turn (presenter view)", expanded=False):
        icon_schema = "✓" if audit.get("schema_used") else "–"
        icon_rag = "✓" if audit.get("rag_used") else "–"
        cls_schema = "on" if audit.get("schema_used") else "off"
        cls_rag = "on" if audit.get("rag_used") else "off"
        st.markdown(
            f'<span class="cbt-badge {cls_schema}">{icon_schema} Schema grounded</span>'
            f'<span class="cbt-badge {cls_rag}">{icon_rag} SNOMED grounded</span>',
            unsafe_allow_html=True,
        )

        if schema:
            st.markdown("**Client schema extracted**")
            for bucket, label in [
                ("triggers", "Triggers"),
                ("automatic_thoughts", "Automatic thoughts"),
                ("emotions", "Emotions"),
                ("behaviors", "Behaviors"),
            ]:
                items = [i for i in (schema.get(bucket) or []) if isinstance(i, str) and i.strip()]
                if items:
                    st.markdown(f"- **{label}:** " + ", ".join(items))
        else:
            st.caption("No structured schema extracted for this turn.")

        if rag and rag.get("concepts"):
            st.markdown("**Retrieved clinical concepts**")
            chips = "".join(
                f'<span class="cbt-chip">{html_lib.escape(c["term"])}</span>'
                for c in rag["concepts"]
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("No SNOMED concepts retrieved for this turn.")

st.markdown(
    '<div class="cbt-footer-note">Research prototype — not a substitute for professional mental health care.</div>',
    unsafe_allow_html=True,
)
