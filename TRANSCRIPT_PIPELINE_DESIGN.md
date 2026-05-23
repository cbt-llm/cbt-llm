# Transcript-Based Pipeline Design Document

## Overview

This document describes all design decisions, prompt changes, and architectural
changes required to transition the CBT-LLM pipeline from LLM-generated patient
turns (seed-based) to fixed real-world transcript turns.
---

## 1. What Is Changing and Why

### Before (current)
- A human-written **seed sentence** (e.g. "I feel overwhelmed at work and shut down.")
  is passed as the first patient message.
- A **core issue label** (e.g. "anger and emotional dysregulation") is also passed
  at the start and injected into the therapist prompt as a hard anchor.
- The **patient LLM (GPT-4o-mini)** generates all subsequent patient turns by
  simulating a CBT patient persona.
- The therapist is told: "the patient came in with THIS concern — always anchor back to it."

### After (new design)
- **Real transcripts** from two datasets drive the patient side:
  - **RealCBT split files**: `Real_cbt_split/file_N_client.txt`
    Each line: `client_turn_N: <text>`
  - **ESConv client JSON**: `ESConv_client.json`
    Each entry has a `turn` label and `content` field.
- The patient LLM generation is **commented out** (kept but inactive).
- There is **no pre-supplied core issue** or seed sentence.
- The therapist must **infer the core concern from the unfolding conversation**
  and use that to anchor responses when the client drifts.
- After the full session, the therapist LLM is asked to **tag the core issue**
  from the transcript.
- Each turn in the output is tagged with a **time label** (T0, T1, T2, ...).

---

## 2. File-by-File Changes

### 2.1 `src/cbt_llm/cbt_mcot.py`

#### A. `build_anchor_block(seed, patient_history)` → `build_anchor_block(patient_history)`

**Why:** We no longer have a fixed seed. The anchor must be inferred dynamically
from what the client has said so far.

**Old behaviour:**
- Returned `""` if seed or history was empty.
- Injected the literal seed sentence + recent history into the protocol prompt.
- Told the model "anchor to THIS original concern."

**New behaviour:**
- If `patient_history` is empty → **INTRODUCTORY TURN block**: tells the model
  this is the first message — there is no prior history yet. The model must
  read the actual client message and branch:
  - Client opens with a **greeting / small talk** ("Hello", "Hi, how are you") →
    respond warmly and briefly; invite them to share what brought them in.
  - Client opens **directly with their concern** ("I've been really anxious...",
    "I had a terrible week at work...") → engage therapeutically right away using
    this protocol's technique; do NOT waste a turn on pleasantries.
  This distinction is critical: the introductory block must not force a warm
  greeting response when the client has already opened with substance.
- If history exists → **SESSION CONTEXT block**:
  - Shows the last ≤6 client turns.
  - Instructs the model to **infer the core concern silently** from history.
  - Classifies the current message into three types and tells the model how to handle each:
    - **GREETING / SMALL TALK** → respond warmly, brief, transition naturally.
    - **ON TOPIC** → apply the protocol technique directly.
    - **DISTRACTION / TANGENT** → (1) short acknowledgment, (2) gentle bridge phrase,
      (3) apply protocol to inferred core concern.
  - Provides labelled examples for each type.

**Decision:** Use last 6 turns (not 4) so the model has richer context to infer
the concern from longer transcripts with slow reveals.

---

#### B. `generate_candidate(...)` — update `build_anchor_block` call

Change:
```python
anchor_block = build_anchor_block(seed, patient_history)
```
to:
```python
anchor_block = build_anchor_block(patient_history)
```

`seed` parameter remains in the function signature for backward compatibility
but is no longer passed to `build_anchor_block`.

---

#### C. `evaluate_candidates(...)` — remove hard seed filter

**Old behaviour:**
- Built a `seed_block` string injecting the fixed seed as an "ANCHORING FILTER
  (READ FIRST — OVERRIDES PROTOCOL FIT)".
- This penalised candidates that drifted from the seed, even if the client had
  moved on naturally.

**New behaviour:**
- Remove `seed_block`.
- Add a **GREETING AWARENESS note**: if the client message looks like a greeting
  (short, contains "hello"/"hi"/"hey"/"how are"), add a note to prefer the
  candidate that responds naturally rather than with immediate clinical probing.
- Judge still selects the best protocol based on clinical fit and session context.
- `seed=None` parameter kept in the signature for backward compat but unused.

---

#### D. `mcot_therapist_reply(...)` — seed parameter comment

`seed=None` kept in signature. Add inline comment:
```python
seed=None,  # kept for backward compat — core concern now inferred from history
```

No functional change inside; `build_anchor_block` and `evaluate_candidates`
already updated to not use it.

---

#### E. NEW function: `generate_session_core_issue(llm, transcript)`

**Purpose:** After the full session loop finishes, call the therapist LLM once
more to analyse the whole conversation and produce a short core issue label.

**Input:** The `transcript` list (list of turn dicts with `patient.query` and
`llm_response.response`).

**Prompt:**
```
You are a clinical supervisor reviewing a therapy session transcript.
Based on the full conversation, identify the client's core psychological issue
in 3–8 words.
Output ONLY the core issue label — nothing else, no punctuation.
Examples: "anger and emotional dysregulation",
          "anxiety about social situations",
          "grief and loss of identity"
```

**Temperature:** 0.0 (deterministic label)
**max tokens:** 50

**Output:** A short string, e.g. `"trauma and hypervigilance in public spaces"`

This is added to the output JSON under `metadata.inferred_core_issue`.

---

### 2.2 `src/cbt_llm/multiturn_convo.py`

#### A. `THERAPIST_CBT_MCOT_PROMPT` — full rewrite

**What is removed:**
- The `TOPIC ANCHORING` section at the bottom. It referenced the seed sentence
  and said "DO NOT follow tangents at the expense of the seed topic." This no
  longer applies since there is no seed.

**What is improved / added:**

1. **Framing**: "skilled CBT therapist in an active therapy session" rather than
   "agent." Signals natural, human therapeutic tone.

2. **COGNITIVE MODEL section**: Unchanged in structure, clarified in wording.
   The model should silently identify which element (Trigger, Automatic Thought,
   Emotion, Behavior) is most active in the current message and let that guide
   the depth of response.

3. **CLINICAL CONTEXT section**: Unchanged in structure. Added explicit instruction
   that CONTRADICTION items must not be used even if tempting.

4. **MULTIPLE CHAIN-OF-THOUGHT REASONING section**: Unchanged. Clarify that the
   three candidates are internally simulated and the best one selected.

5. **NEW — RESPONSE GUIDANCE section** (replaces TOPIC ANCHORING):
   - **Introductory / greeting turns**: respond warmly, do not force clinical
     questions, let the opening feel human and safe.
   - **Distraction / tangent turns**: acknowledge briefly, bridge back to inferred
     core concern using natural language (not a rigid formula).
   - **On-topic turns**: apply the selected CBT protocol directly.
   - **General rules**: same as before (no diagnosis, no CBT jargon, no hidden
     context reveal, 2–4 sentences, one question at a time).

---

#### B. `THERAPIST_CBT_PROMPT` — improvements

Same structural improvements as the MCOT prompt:
- Remove any seed-referencing language.
- Add **RESPONSE GUIDANCE** section with greeting/distraction/on-topic handling.
- Improve instruction that the model should reason from schema + RAG data, not
  just acknowledge them.
- Add: "Before selecting a protocol, consider what the client most needs right
  now: to feel heard, to examine a belief, or to find a broader view."

---

#### C. NEW function: `load_client_transcript(path: str) -> List[str]`

Loads ordered client utterances from either format:

**RealCBT `.txt`:**
```
client_turn_1: <text>
client_turn_2: <text>
...
```
Parsed with: `re.match(r'^client_turn_\d+:\s*(.*)', line)`

**ESConv client `.json`:**
```json
[
  {
    "experience_type": "Current Experience",
    "emotion_type": "anger",
    "problem_type": "problems with friends",
    "situation": "I have complete unsupportive friends...",
    "survey_score": { ... },
    "dialog": [
      { "turn": "client_turn_1", "annotation": {}, "content": "Hello\n" },
      ...
    ]
  },
  ...   ← 1300 entries total, each is its own transcript
]
```

**Each entry in ESConv is treated as a separate transcript and produces its own
output file.** Parsed by extracting `entry["content"].strip()` from
`data[index]["dialog"]` where `index` = `--transcript_index`.

The ESConv metadata (`emotion_type`, `problem_type`, `situation`) is carried
into the output JSON `metadata` block so each file is self-documenting.

`load_client_transcript(path, index=0)` accepts an `index` parameter:
- For `.txt` files: `index` is ignored (one transcript per file).
- For `.json` files: selects `data[index]`.

Returns a `List[str]` of utterances in order.

---

#### D. `run_session(...)` — structural changes

**Signature additions:**
```python
transcript_source: str = None,   # path to client .txt or .json file
transcript_index: int = 0,       # which ESConv conversation to use (JSON only)
```

**Commented out (not removed):**
```python
# patient_llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# patient_chat = [{"role": "system", "content": PATIENT_SYSTEM.format()}]
# last_patient = seed    ← replaced by transcript loading
```

**New pre-loop block:**
```python
client_turns = load_client_transcript(transcript_source) if transcript_source else []
max_turns = min(turns, len(client_turns)) if client_turns else turns
last_patient = client_turns[0] if client_turns else (seed or "")
```

**Loop change:**
```python
for turn_idx in range(max_turns):
    if client_turns:
        last_patient = client_turns[turn_idx]   # load from transcript each turn
```

**Commented out — seed reminder injection** (in `cbt_mcot` branch):
```python
# if turn_idx > 0:
#     augmented_patient = (
#         f"[REMINDER — the user came in with this concern at the start of session: \"{seed}\". "
#         f"Stay anchored to it.]\n\n"
#         f"Patient's current message: {last_patient}"
#     )
# else:
#     augmented_patient = last_patient
augmented_patient = last_patient   # always use the raw client turn
```

**Commented out — patient LLM generation block:**
```python
# patient_resp = patient_llm.chat.completions.create(
#     model=patient_model,
#     messages=patient_chat + [{"role": "user", "content": therapist_reply}],
#     temperature=0.9,
#     max_tokens=100,
# )
# patient_reply = patient_resp.choices[0].message.content.strip()
#
# if turn_idx in distraction_map:
#     patient_reply = distraction_map[turn_idx]
# elif looks_like_patient_drift(patient_reply):
#     patient_reply = patient_llm.chat.completions.create(...).choices[0]...
#
# patient_chat.append({"role": "assistant", "content": patient_reply})
```

**End of loop — patient tracking:**
```python
# last_patient = patient_reply        # commented out — transcript drives turns
patient_history.append(last_patient)  # still track for anchor reasoning
```

**Turn record — add time tag:**
```python
turn_record = {
    "turn": turn_idx,
    "time_tag": f"T{turn_idx}",        # NEW
    "patient": { ... },
    "llm_response": therapist_block
}
```

**Post-loop — core issue inference:**
```python
inferred_core_issue = generate_session_core_issue(therapist_llm, transcript)
```

Added to output metadata:
```python
"metadata": {
    ...
    "transcript_source": transcript_source,     # NEW
    "inferred_core_issue": inferred_core_issue, # NEW (replaces pre-supplied core_issue)
    # "seed": seed,          # commented out
    # "core_issue": core_issue,  # commented out
    ...
}
```

---

#### E. Import update

```python
from cbt_llm.cbt_mcot import mcot_therapist_reply, CBT_PROTOCOLS, generate_session_core_issue
```

---

#### F. `main()` — argument changes

**Added:**
```python
ap.add_argument("--transcript_source", help="Path to client .txt or ESConv client .json")
ap.add_argument("--transcript_index", type=int, default=0,
                help="Which conversation to use from an ESConv JSON file")
```

**Commented out (kept):**
```python
# ap.add_argument("--seed", required=True)
ap.add_argument("--seed", default=None)        # now optional / unused
# ap.add_argument("--core_issue")
ap.add_argument("--core_issue", default=None)  # now optional / unused
```

---

### 2.3 `run_experiments.sh` — structural changes

**Commented out:**
```bash
# seeds=(
#   "I feel calm most of the time ..."
#   ...
# )

# core_issues=(
#   "anger and emotional dysregulation"
#   ...
# )
```

**New — RealCBT transcript loop:**
```bash
REALCBT_DIR="/Users/poojaguttal/Downloads/realcbt-dataset-main/RealCBT_Dataset/Real_cbt_split"

mapfile -t client_files < <(ls "${REALCBT_DIR}"/*_client.txt | sort -V)

for i in "${!client_files[@]}"; do
  run=$((i+1))
  transcript_file="${client_files[$i]}"
  fname=$(basename "$transcript_file" _client.txt)

  CMD=(
    python -m cbt_llm.multiturn_convo
    --therapist_model "$MODEL"
    --therapist_mode "$MODE"
    --turns "$TURNS"
    --k "$K"
    --transcript_source "$transcript_file"
    # --seed "$seed"        # commented out — turns come from transcript
    # --core_issue "$issue" # commented out — inferred post-session
    --transcript_json "${OUTDIR}/realcbt_${MODE}_${fname}.json"
  )
  ...
done
```

**New — ESConv transcript loop (separate):**
Each of the 1300 entries in `ESConv_client.json` is its own transcript.
```bash
ESCONV_JSON="/Users/poojaguttal/Downloads/ESConv_client.json"
ESCONV_COUNT=$(python3 -c "import json; print(len(json.load(open('${ESCONV_JSON}'))))")

for i in $(seq 0 $((ESCONV_COUNT - 1))); do
  run=$((i+1))

  CMD=(
    python -m cbt_llm.multiturn_convo
    --therapist_model "$MODEL"
    --therapist_mode "$MODE"
    --turns "$TURNS"
    --k "$K"
    --transcript_source "$ESCONV_JSON"
    --transcript_index "$i"
    --transcript_json "${OUTDIR}/esconv_${MODE}_transcript_${run}.json"
  )
  ...
done
```

The `DATASET` argument (new, e.g. `realcbt` or `esconv`) controls which loop runs.

**Distractions:** The distraction schedule (random / thoughtful) is kept but
commented out in the CMD build — with real transcripts the client turns are
fixed and injection would corrupt the data. Left in place so it can be
re-enabled for synthetic ablation experiments.

---

## 3. Output JSON Format (after changes)

### RealCBT output file
```json
{
  "metadata": {
    "llm_response": "mistral:7b",
    "patient_model": "gpt-4o-mini",
    "mode": "cbt_mcot",
    "turns": 12,
    "dataset": "realcbt",
    "transcript_source": "Real_cbt_split/file_9_client.txt",
    "transcript_index": null,
    "esconv_meta": null,
    "inferred_core_issue": "trauma and hypervigilance in public spaces",
    "distraction_schedule": null,
    "intervention_flags": {
      "use_schema": true,
      "use_rag": true,
      "use_protocol": true
    }
  },
  "transcript": [ ... ]
}
```

### ESConv output file
Each ESConv entry produces one output file. The `esconv_meta` block carries the
source conversation's context so the file is self-documenting.
```json
{
  "metadata": {
    "llm_response": "mistral:7b",
    "patient_model": "gpt-4o-mini",
    "mode": "cbt_mcot",
    "turns": 8,
    "dataset": "esconv",
    "transcript_source": "ESConv_client.json",
    "transcript_index": 42,
    "esconv_meta": {
      "experience_type": "Current Experience",
      "emotion_type": "anger",
      "problem_type": "problems with friends",
      "situation": "I have complete unsupportive friends its to the point where i dont even feel like i have friends any more."
    },
    "inferred_core_issue": "loneliness and unsupportive social relationships",
    "distraction_schedule": null,
    "intervention_flags": {
      "use_schema": true,
      "use_rag": true,
      "use_protocol": true
    }
  },
  "transcript": [
    {
      "turn": 0,
      "time_tag": "T0",
      "patient": {
        "role": "patient",
        "query": "Hello",
        "schema": { ... },
        "retrieval": { ... },
        "distraction_injected": false
      },
      "llm_response": {
        "role": "cbt_agent",
        "response": "Hello, good to see you today. What's been on your mind?",
        "protocol_used": "validate_and_reflect",
        "mcot_candidates": { ... }
      }
    },
    ...
  ]
}
```

---

## 4. What Is NOT Changed

- `PATIENT_SYSTEM` prompt — kept (still usable if LLM patient is re-enabled)
- `THERAPIST_BASELINE_PROMPT` — kept as-is
- `OllamaChat` / `OpenAIChat` classes — unchanged
- `FindingsPipeline`, `retrieve_snomed_matches`, `extract_user_schema` — unchanged
- `audit_grounding`, `classify_reasoning_concepts`, `classify_transcript` — unchanged
- `detect_anchor_decision` — kept (still useful for post-hoc analysis even with
  real transcripts, though `seed` will be `None`)
- `parse_gpt_oss_output` — unchanged
- Safety signal detection (`_is_safety_signal`) — unchanged
- `evaluate_candidates` judge logic and protocol descriptions — unchanged in
  structure, only seed_block removed

---

## 5. Decision Log

| Decision | Rationale |
|---|---|
| Comment out patient LLM, not delete | Allows easy re-enable for synthetic ablation experiments later |
| Infer core concern from history, not seed | Matches real therapy: therapist learns the concern from what the client says, not a pre-brief |
| Use last 6 turns in anchor block | Longer transcripts have slow reveals; 4 was too short for complex cases |
| Greeting detection by message length + keywords | Simple and fast; avoids extra LLM call at session start |
| `generate_session_core_issue` post-session (not per-turn) | Core issue only becomes clear after the full session; per-turn labelling would be premature |
| `time_tag: "T{turn_idx}"` | Lightweight, easy to filter/slice in evaluation later |
| `transcript_source` in metadata output | Makes every output JSON self-documenting — you can trace which real transcript it came from |
| Keep `--seed` and `--core_issue` args (default None) | Backward compat — old shell scripts won't break |
| `sort -V` for client file iteration | Natural sort: file_2 before file_10 |
