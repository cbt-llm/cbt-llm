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


_SAFETY_SIGNALS = [
    "not want to wake up", "don't want to wake up", "end my life", "ending my life",
    "hurt myself", "hurting myself", "kill myself", "killing myself",
    "not be here", "disappear forever", "die alone", "better off dead",
    "no reason to live", "can't go on", "want to die", "wish i was dead",
]


def _is_safety_signal(text: str) -> bool:
    t = text.lower()
    return any(sig in t for sig in _SAFETY_SIGNALS)


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


def build_anchor_block(patient_history):
    """
    Dynamic anchor block injected into each candidate's principle prompt.
    Infers the client's core concern from accumulated session history — no fixed seed required.
    On turn 0 (empty history), guides the model to read the client's actual message
    and decide whether it is a greeting or a direct opening of their concern.
    """
    if not patient_history:
        return """
━━━━━━━━━━━━━━━━━━━
FIRST MESSAGE — READ THE CLIENT CAREFULLY
━━━━━━━━━━━━━━━━━━━

This is the client's very first message. There is no prior session history yet.

Read the client's message and decide:

A) GREETING / SMALL TALK — the client says hello, asks how you are, or opens
   with pleasantries ("Hi", "Hello", "Good morning", "How are you?").
   → Respond warmly and briefly. Invite them to share what brought them in today.
   → Do NOT ask a clinical question yet. Keep it to 1–2 sentences.
   → Example: "Hello, good to see you. What brings you in today?"

B) DIRECT OPENING — the client immediately shares their concern, a situation,
   a feeling, or an experience they want to talk about.
   → Engage therapeutically from the very first response. Apply this protocol's
     technique directly. Do NOT waste the turn on pleasantries.
   → Example (client: "I've been really anxious about my job for weeks"):
     "It sounds like that worry has been sitting with you for a while.
      What does the anxiety feel like when it shows up most strongly?"

Match your response to what the client actually said.
""".strip()

    history_lines = "\n".join(
        f'  Turn {i + 1}: "{msg[:150]}{"..." if len(msg) > 150 else ""}"'
        for i, msg in enumerate(patient_history[-6:])
    )

    return f"""
━━━━━━━━━━━━━━━━━━━
SESSION CONTEXT — DYNAMIC TOPIC AWARENESS
━━━━━━━━━━━━━━━━━━━

What the client has shared so far this session:
{history_lines}

STEP 1 — INFER CORE CONCERN:
Based on the session history above, determine what the client's primary reason
for being in therapy appears to be. This is your inferred core concern.
Do not state it aloud — use it silently to anchor your response.

STEP 2 — CLASSIFY THE CURRENT MESSAGE AND RESPOND ACCORDINGLY:

A) GREETING / SMALL TALK — client says hello, asks how you are, or makes
   casual conversation unrelated to their concern.
   → Respond warmly and briefly. Naturally transition toward the therapeutic topic.
   → Do NOT immediately ask a clinical question.

B) ON TOPIC — client continues discussing their core concern or a directly
   related theme.
   → Apply this protocol's technique directly and therapeutically.

C) DISTRACTION / TANGENT — client shifts to an unrelated topic, story, or
   surface-level event that has nothing to do with their core concern.
   → Do all three IN ORDER:
      (1) ONE short clause (≤10 words) acknowledging what they just said.
      (2) A gentle bridge: "I'd like to come back to what you've been working
          through..." or "That connects to what you mentioned earlier about..."
      (3) Apply this protocol's technique toward the inferred core concern.

━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━

EXAMPLE 1 (greeting mid-session → warm + transition):
Client: "Hey, how are you doing today?"
GOOD: "I'm doing well, thanks for asking. How have things been since we last spoke?"

EXAMPLE 2 (distraction → BRIDGE):
Inferred concern: client struggles with anger at work.
Client now: "I watched this really funny show last night, completely zoned out."
BAD: "That sounds like a nice escape! What did you enjoy about it?"
GOOD: "Glad you got some downtime. I do want to come back to what you've been
working through — when did that frustration at work last show up for you?"

EXAMPLE 3 (on topic → direct response):
Inferred concern: client struggles with anger at work.
Client now: "I snapped at my coworker again yesterday."
GOOD: "That same pattern came up again. What was going through your mind right
before you reacted?"

Apply the same classification and logic to the actual conversation below.
""".strip()


def generate_candidate(llm, base_prompt, hidden_context, patient_text, protocol,
                       seed=None, turn_idx=0, patient_history=None):

    anchor_block = build_anchor_block(patient_history)

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
                        seed=None, patient_history=None):  # seed unused — core concern inferred from history

    greeting_note = ""
    _greeting_signals = {"hello", "hi", "hey", "how are", "good morning", "good afternoon", "good evening"}
    if patient_text and len(patient_text.strip()) < 60:
        lowered = patient_text.lower()
        if any(g in lowered for g in _greeting_signals):
            greeting_note = """

    ━━━━━━━━━━━━━━━━━━━
    GREETING / OPENING NOTE
    ━━━━━━━━━━━━━━━━━━━

    The client's message appears to be a greeting or social opening.
    Prefer the candidate that responds warmly and naturally.
    A response that immediately launches into clinical questioning is WRONG here.
    The right response acknowledges the client and invites them to share or continue.
    """.rstrip()

    judge_prompt = f"""
    You are evaluating therapist responses in a Cognitive Behavioral Therapy (CBT) conversation.

    Each response was generated using a different CBT intervention protocol.

    Your task: select the response that would most effectively move the therapy forward.
    {greeting_note}

    Step 1 — Read the client message carefully. Identify:
    - Is this a greeting or social pleasantry? If so, see the GREETING NOTE above.
    - Is the client primarily expressing emotion and needing to feel heard?
    - Is the client articulating a specific belief or thought that could be examined?
    - Is the client stuck in a rigid interpretation that a broader view could help shift?

    Step 2 — Match the clinical need to the protocol:
    - validate_and_reflect: client needs emotional acknowledgment BEFORE they can
      examine beliefs. Use early in rapport-building, when distress is high, or
      when the client is not yet ready to reflect.
    - socratic_questioning: client has articulated a specific belief or assumption
      that can be tested with evidence. Use when the client is stable enough to
      reflect and there is a clear belief to examine.
    - alternative_perspective: client is locked into one way of seeing a situation
      and a broader view would reduce distress. Use when the client already feels
      heard and is repeating the same interpretation without movement.

    Step 3 — Has the client already received validation in recent turns? If so,
    continuing to validate may not move the conversation forward. Prefer
    socratic_questioning or alternative_perspective if the client is ready.

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
    seed=None,          # kept for backward compat — core concern now inferred from history
    turn_idx=0,
    patient_history=None,
):
    # Safety signals bypass MCOT entirely — mandatory, no candidate voting
    if _is_safety_signal(patient_text):
        safety_protocol = CBT_PROTOCOLS.get("safety_check")
        if safety_protocol:
            candidate = generate_candidate(
                llm, base_prompt, hidden_context, patient_text,
                safety_protocol, seed=seed, turn_idx=turn_idx,
                patient_history=patient_history,
            )
            return (
                candidate["response"],
                "safety_check",
                {"safety_check": candidate},
                candidate["reasoning"],
            )

    candidate_list = []
    candidate_map = {}

    # Exclude safety_check from normal MCOT candidates
    protocols = [p for p in CBT_PROTOCOLS.values() if p["name"] != "safety_check"]

    for protocol in protocols:

        candidate = generate_candidate(
            llm,
            base_prompt,
            hidden_context,
            patient_text,
            protocol,
            seed=seed,
            turn_idx=turn_idx,
            patient_history=patient_history,
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
        patient_history=patient_history,
    )

    best_protocol = protocols[best_idx]["name"]

    best_response = candidate_map[best_protocol]["response"]
    best_reasoning = candidate_map[best_protocol]["reasoning"]

    return best_response, best_protocol, candidate_map, best_reasoning


def generate_session_core_issue(llm, transcript) -> str:
    """
    Post-session inference: analyses the full transcript and returns a short
    label (3–8 words) describing the client's core psychological issue.
    Called once after the session loop completes.
    """
    conversation_text = "\n".join(
        f"Client: {turn['patient']['query']}\nTherapist: {turn['llm_response']['response']}"
        for turn in transcript
        if turn.get("patient") and turn.get("llm_response")
    )

    if not conversation_text.strip():
        return "unknown"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a clinical supervisor reviewing a therapy session transcript.\n"
                "Based on the full conversation, identify the client's core psychological issue "
                "in 3–8 words.\n"
                "Output ONLY the core issue label — no punctuation, no explanation, nothing else.\n"
                "Examples:\n"
                "  anger and emotional dysregulation\n"
                "  anxiety about social situations\n"
                "  grief and loss of identity\n"
                "  trauma and hypervigilance in public spaces\n"
                "  loneliness and unsupportive social relationships"
            )
        },
        {
            "role": "user",
            "content": f"Session transcript:\n{conversation_text}\n\nCore issue:"
        }
    ]

    try:
        return llm.chat(messages, temperature=0.0, num_predict=50).strip()
    except Exception:
        return "unknown"