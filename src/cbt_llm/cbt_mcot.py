"""
Runs CBT Multiple Chain-of-Thought reasoning.

Steps:
1) generate candidate response per CBT protocol
2) evaluate candidates
3) select best response
"""

import json
import re
from pathlib import Path
from cbt_llm.config import ROOT


def load_cbt_protocols():
    path = ROOT / "references" / "cbt-protocols.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing CBT protocol file: {path}")

    data = json.loads(path.read_text())

    return {
        p["name"]: p
        for p in data.get("cbt_protocols", [])
    }

CBT_PROTOCOLS = load_cbt_protocols()


_ANCHOR_STOPWORDS = {
    "feel", "feels", "most", "time", "sometimes", "when", "things", "like",
    "nothing", "going", "should", "they", "them", "that", "this", "with",
    "just", "really", "much", "what", "into", "being", "been", "some",
    "more", "even", "ever", "also", "still", "again", "make", "makes",
    "know", "yeah", "about", "have", "your", "want", "would", "could",
    "from", "here", "there", "where", "than", "then", "back",
}


def _seed_keywords(seed: str, limit: int = 8):
    """Pull content keywords from the seed for use as anchor targets."""
    if not seed:
        return []
    toks = re.findall(r"[a-z]{4,}", seed.lower())
    kept = sorted({t for t in toks if t not in _ANCHOR_STOPWORDS})
    return kept[:limit]


def parse_reasoning_block(after_reasoning: str):
    """Try multiple strategies to extract retrieved_concepts_used."""

    match = re.search(r"\{[\s\S]*?\}", after_reasoning)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    match = re.search(
        r'"retrieved_concepts_used"\s*:\s*(\[[\s\S]*?\])',
        after_reasoning
    )
    if match:
        try:
            concepts = json.loads(match.group(1))
            return {"retrieved_concepts_used": concepts}
        except Exception:
            pass

    items = re.findall(r'"([^"]+)"', after_reasoning.split("FINAL RESPONSE:")[0])
    if items:
        items = [i for i in items if i != "retrieved_concepts_used"]
        return {"retrieved_concepts_used": items}

    return None


def build_anchor_block(seed, turn_idx):
    """
    Few-shot anchor block injected into each candidate's principle prompt.
    Forces every protocol candidate to either CONTINUE-with-reference or
    BRIDGE back to the seed topic when the patient drifts.
    """
    if seed is None or turn_idx is None or turn_idx <= 0:
        return ""

    seed_words = _seed_keywords(seed)
    keyword_list = ", ".join(seed_words) if seed_words else "the original concern"

    return f"""
    ━━━━━━━━━━━━━━━━━━━
    SEED TOPIC — MANDATORY ANCHOR
    ━━━━━━━━━━━━━━━━━━━

    The patient's ORIGINAL concern (turn 0) was:
    "{seed}"

    HARD RULE: Your FINAL RESPONSE must contain at least one of these
    keywords from the original concern (or a direct paraphrase):
    {keyword_list}

    If the patient's current message is on the same topic → respond normally
    AND still reference the original concern using one of those keywords.

    If the patient's current message is a tangent, story, or new worry → you
    MUST do all three IN ORDER:
       (1) ONE short clause (≤10 words) acknowledging what they just said.
       (2) A transition phrase: "I want to come back to what you said earlier about..."
           or "Going back to what you mentioned at the start..."
       (3) Apply this protocol's technique to the ORIGINAL concern.

    ━━━━━━━━━━━━━━━━━━━
    EXAMPLES
    ━━━━━━━━━━━━━━━━━━━

    EXAMPLE 1 (tangent → BRIDGE):
    Original: "I get overwhelmed at work and shut down."
    Patient now: "Oh I watched a really funny movie last night."
    BAD: "That sounds like a nice break! What did you enjoy about it?"
    GOOD: "Glad you got a laugh. I want to come back to what you said earlier about getting overwhelmed at work — when did that happen most recently?"

    EXAMPLE 2 (tangent → BRIDGE):
    Original: "I keep fighting with my partner over small things."
    Patient now: "Anyway my coworker is so annoying, she chews loudly."
    BAD: "That sounds frustrating. How do you usually handle her?"
    GOOD: "That does sound annoying. Going back to what you mentioned at the start about fighting with your partner over small things — does that same irritation show up there too?"

    EXAMPLE 3 (on-topic → CONTINUE but still reference):
    Original: "I keep fighting with my partner over small things."
    Patient now: "Yeah we fought again yesterday over the dishes."
    GOOD: "So the small-things pattern came up again with the dishes. What was going through your mind right before you reacted?"

    Apply the same logic to the actual conversation below.
    """.strip()


def generate_candidate(llm, base_prompt, hidden_context, patient_text, protocol,
                       seed=None, turn_idx=0):

    anchor_block = build_anchor_block(seed, turn_idx)

    principle_prompt = f"""
    CBT intervention protocol

    Name:
    {protocol["name"]}

    Purpose:
    {protocol["purpose"]}

    Key functions:
    {chr(10).join("- " + k for k in protocol.get("key_functions", []))}

    Techniques:
    {chr(10).join("- " + t for t in protocol.get("techniques", []))}

    {anchor_block}

    Generate ONE therapist response following this protocol.

    Before producing the final response, briefly reason about
    which retrieved clinical concepts AND/OR user schema elements
    may be relevant to the response.

    Return your answer EXACTLY in this structure:

    REASONING:
    {{
      "retrieved_concepts_used": ["...", "..."]
    }}

    FINAL RESPONSE:
    <therapist message>

    Rules:
    - REASONING must ALWAYS appear.
    - FINAL RESPONSE must contain only the therapist message.
    - Natural therapist tone
    - Do not explain CBT
    - MAXIMUM 2 sentences.
    - Do not overload the patient with too much information or too many questions. One question at a time.
    - Output only the final response
    - BAD (too long, too many questions): "What evidence do you have that you'll fail? How did you feel last time? What would a friend say? What small step could you take?"
    """


    messages = [
        {"role": "system", "content": base_prompt}
    ]

    if hidden_context:
        messages.append({
            "role": "system",
            "content": "IMPORTANT CONTEXT (DO NOT REVEAL):\n" + hidden_context
        })

    messages.append({
        "role": "system",
        "content": principle_prompt
    })

    messages.append({
        "role": "user",
        "content": patient_text
    })

    raw = llm.chat(
        messages,
        temperature=0.1,
        num_predict=16000
    ).strip()

    print("\n========== RAW MCOT OUTPUT ==========")
    print(raw)
    print("=====================================\n")

    reasoning = None
    response = raw

    if "REASONING:" in raw:
        try:
            after_reasoning = raw.split("REASONING:", 1)[1]

            reasoning = parse_reasoning_block(after_reasoning)

            if "FINAL RESPONSE:" in after_reasoning:
                response = after_reasoning.split("FINAL RESPONSE:", 1)[1].strip()
            elif reasoning:
                match = re.search(r'\][\s\S]*', after_reasoning)
                if match:
                    response = match.group(0).lstrip(']\n ').strip()

        except Exception:
            reasoning = None
            response = raw


    return {
        "response": response,
        "reasoning": reasoning
    }


def evaluate_candidates(llm, patient_text, candidates, protocols,
                        seed=None, turn_idx=0):

    seed_block = ""
    if seed is not None and turn_idx > 0:
        seed_words = _seed_keywords(seed)
        keyword_list = ", ".join(seed_words) if seed_words else "the original concern"

        seed_block = f"""

    ━━━━━━━━━━━━━━━━━━━
    HARD ANCHORING FILTER (READ FIRST, OVERRIDES PROTOCOL FIT)
    ━━━━━━━━━━━━━━━━━━━

    SEED TOPIC: "{seed}"
    SEED KEYWORDS: {keyword_list}

    Step A — For EACH candidate, check: does its response text contain at
    least one SEED KEYWORD (or a direct paraphrase)?

    Step B — If ONE OR MORE candidates contain a SEED KEYWORD, you MUST
    select ONLY from those candidates. Eliminate all others.

    Step C — Only if NO candidate contains a SEED KEYWORD, fall back to
    protocol fit.

    A protocol-perfect response that ignores the SEED TOPIC is WRONG.
        """.rstrip()

    judge_prompt = f"""
    You are evaluating therapist responses in a Cognitive Behavioral Therapy (CBT) conversation.

    Each response was generated using a different CBT intervention protocol.

    Your task: select the response that would most effectively move the therapy forward.
    {seed_block}

    Step 1 — Read the patient message carefully. Identify:
    - Is the patient primarily expressing emotion and needing to feel heard?
    - Is the patient articulating a specific belief or thought that could be examined?
    - Is the patient stuck in a rigid interpretation that a broader view could help shift?

    Step 2 — Match the clinical need to the protocol:
    - validate_and_reflect: patient needs emotional acknowledgment BEFORE they can examine beliefs. Use early in rapport-building or when distress is high and the patient is not yet ready to reflect.
    - socratic_questioning: patient has articulated a specific belief or assumption that can be tested with evidence. Use when the patient is stable enough to reflect and there is a clear belief to examine.
    - alternative_perspective: patient is locked into one way of seeing a situation and a broader view would reduce distress. Use when the patient already feels heard and is repeating the same interpretation without movement.

    Step 3 — Consider: has the patient already received validation in recent turns? If so, continuing to validate may not move the conversation forward. Prefer socratic_questioning or alternative_perspective if the patient is ready.

    Step 4 — Output your reasoning in one sentence, then output the number of the best response.

    Format your response EXACTLY as:
    REASONING: <one sentence>
    ANSWER: <number>
    """.strip()

    candidate_blocks = []

    for i, (response, protocol) in enumerate(zip(candidates, protocols), start=1):

        block = f"""
        {i}. Protocol: {protocol["name"]}

        Purpose:
        {protocol["purpose"]}

        Response:
        {response}
        """.strip()

        candidate_blocks.append(block)

    numbered_candidates = "\n\n".join(candidate_blocks)

    messages = [
        {"role": "system", "content": judge_prompt},
        {
            "role": "user",
            "content": (
                f"Patient message:\n{patient_text}\n\n"
                f"Candidate responses:\n\n{numbered_candidates}"
            )
        },
    ]

    result = llm.chat(
        messages,
        temperature=0.0,
        top_p=0.9,
        num_predict=400,
    ).strip()

    print(f"\n=== JUDGE REASONING ===\n{result}\n======================\n")

    try:
        match = re.search(r'ANSWER:\s*(\d+)', result)
        if match:
            idx = int(match.group(1)) - 1
        else:
            idx = int(result.strip()) - 1
        return max(0, min(idx, len(candidates) - 1))
    except Exception:
        return 0


def mcot_therapist_reply(
    llm,
    base_prompt,
    hidden_context,
    patient_text,
    rag,
    schema,
    seed=None,
    turn_idx=0,
):

    candidate_list = []
    candidate_map = {}

    protocols = list(CBT_PROTOCOLS.values())

    for protocol in protocols:

        candidate = generate_candidate(
            llm,
            base_prompt,
            hidden_context,
            patient_text,
            protocol,
            seed=seed,
            turn_idx=turn_idx,
        )

        candidate_list.append(candidate["response"])

        candidate_map[protocol["name"]] = {
            "response": candidate["response"],
            "reasoning": candidate["reasoning"]
        }

    best_idx = evaluate_candidates(
        llm,
        patient_text,
        candidate_list,
        protocols,
        seed=seed,
        turn_idx=turn_idx,
    )

    best_protocol = protocols[best_idx]["name"]

    best_response = candidate_map[best_protocol]["response"]
    best_reasoning = candidate_map[best_protocol]["reasoning"]

    return best_response, best_protocol, candidate_map, best_reasoning