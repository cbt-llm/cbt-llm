# model_runner.py

import subprocess
import re
import json
from datetime import datetime
from config import OUTPUT_LLM_DIR
import os


MODELS = [
    "gemma3:12b",
    "llama2:13b",
    "vicuna:13b"
]

PROMPT_TEMPLATE = """
You are assisting with exploratory research using SNOMED CT as the
reference clinical ontology.

Your task:
1. Understand the user text semantically.
2. Perform a conceptual (semantic) search over SNOMED CT
   mental and behavioral health concepts.
3. Select up to 5 relevant SNOMED CT concept names only (no codes).
4. When applicable, prefer higher-level parent concepts
   connected via IS-A relationships.

Rules:
- Do NOT invent or guess medical codes.
- Do NOT diagnose or label the user.
- Only suggest high-level SNOMED CT mental or behavioral health concepts.
- If nothing is relevant, say: "No clear mental health concepts."

User text:
"{user_text}"

Output format (JSON only):
{{
  "concepts": [
    {{
      "term": "SNOMED concept name"
    }}
  ]
}}
"""

def run_model(user_text: str, model_name: str):
    prompt = PROMPT_TEMPLATE.format(user_text=user_text)

    result = subprocess.run(
        ["ollama", "run", model_name],
        input=prompt,
        text=True,
        capture_output=True
    )

    raw_output = result.stdout.strip()

    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not match:
        return {
            "error": f"No valid JSON returned for model {model_name}",
            "raw_output": raw_output
        }

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {
            "error": f"Invalid JSON format for model {model_name}",
            "raw_output": raw_output
        }

def run_all_models(user_text: str):

    results = {}

    print("\n========== Running All Models ==========\n")

    for model in MODELS:
        print(f"  - Running model: {model}")
        model_result = run_model(user_text, model)
        results[model] = model_result

    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"llm_results_{timestamp}.json"
    output_path = os.path.join(OUTPUT_LLM_DIR, output_filename)

    # Prepare structured output
    final_output = {
        "timestamp": timestamp,
        "user_text": user_text,
        "models_run": MODELS,
        "results": results
    }

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nResults saved to:\n{output_path}\n")

    return final_output

