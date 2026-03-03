import os
import re
import json
from datetime import datetime
import pandas as pd
import anthropic

from cbt_llm.config import OUTPUT_LLM_DIR
from .prompt_templates.prompt import PROMPT_SNOMED


client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


CLAUDE_MODELS = [
    "claude-opus-4-5-20251101"
]


def run_model(user_text: str, model_name: str):
    prompt = PROMPT_SNOMED.format(user_text=user_text)

    response = client.messages.create(
        model=model_name,
        max_tokens=1500,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_output = response.content[0].text

    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not match:
        return {"error": "No valid JSON returned", "raw_output": raw_output}

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format", "raw_output": raw_output}


def run_claude_model(user_text_list: list):

    os.makedirs(OUTPUT_LLM_DIR, exist_ok=True)

    for model in CLAUDE_MODELS:

        print(f"\n========== Running Claude Model: {model} ==========\n")
        all_rows = []

        for user_text in user_text_list:

            model_result = run_model(user_text, model)

            row = {"User Query": user_text}

            if "error" in model_result:
                for i in range(1, 6):
                    row[f"Concept {i}"] = model_result["error"]
            else:
                findings = model_result.get("findings", [])
                for i in range(5):
                    if i < len(findings):
                        row[f"Concept {i+1}"] = findings[i].get("term", "")
                    else:
                        row[f"Concept {i+1}"] = ""

            all_rows.append(row)

        df = pd.DataFrame(all_rows)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"claude_{model}_{timestamp}.csv"
        filepath = os.path.join(OUTPUT_LLM_DIR, filename)

        df.to_csv(filepath, index=False)
        print(f"Saved to: {filepath}")