import json
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

def generate_candidate(llm, base_prompt, hidden_context, patient_text, protocol):

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

    Generate ONE therapist response following this protocol.

    Rules:
    - natural therapist tone
    - do not explain CBT
    - 2–4 sentences
    - output only the response
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

    return llm.chat(
        messages,
        temperature=0.4,
        num_predict=400,
    ).strip()

def evaluate_candidates(llm, patient_text, candidates):

    judge_prompt = """
    You are evaluating therapist responses.

    Select the response that best advances therapy.

    Criteria:
    - empathy
    - exploration of beliefs
    - therapeutic usefulness
    - conversational flow

    Return ONLY the number of the best response.
    """

    numbered = "\n\n".join(
        f"{i+1}. {c}" for i, c in enumerate(candidates)
    )

    messages = [
        {"role": "system", "content": judge_prompt},
        {
            "role": "user",
            "content": f"Patient message:\n{patient_text}\n\nResponses:\n{numbered}"
        },
    ]

    result = llm.chat(
        messages,
        temperature=0.0,
        num_predict=20,
    )

    try:
        idx = int(result.strip()) - 1
        return max(0, min(idx, len(candidates) - 1))
    except:
        return 0


def mcot_therapist_reply(
    llm,
    base_prompt,
    hidden_context,
    patient_text
):
    """
    Runs CBT Tree-of-Thought reasoning.

    Steps:
    1) generate candidate response per CBT protocol
    2) evaluate candidates
    3) select best response
    """

    candidates = []
    protocols = list(CBT_PROTOCOLS.values())

    for protocol in protocols:

        candidate = generate_candidate(
            llm,
            base_prompt,
            hidden_context,
            patient_text,
            protocol
        )

        candidates.append(candidate)

    best_idx = evaluate_candidates(
        llm,
        patient_text,
        candidates
    )

    best_response = candidates[best_idx]
    best_protocol = protocols[best_idx]["name"]

    return best_response, best_protocol, candidates