from pathlib import Path
import pandas as pd

ROOT = Path("evals_summary")

JUDGES = ["gpt-5.1", "qwen3-14b"]

MODEL_NAMES = {
    "gemma": "gemma3:12b",
    "deepseek": "deepseek-r1:8b",
    "mistral": "mistral:7b",
    "gpt": "gpt-oss:20b"
}

MODE_NAMES = {
    "baseline": "Baseline",
    "cbt": "CBT Chain-of-Thought",
    "cbt_mcot": "CBT Multiple-Chain-of-Thought"
}

MODE_ORDER = [
    "Baseline",
    "CBT Chain-of-Thought",
    "CBT Multiple-Chain-of-Thought"
]

OUT_FILE = ROOT / "final_results_table.csv"


def load_judge_data(judge):

    rows = []
    judge_dir = ROOT / judge

    for model_dir in judge_dir.iterdir():

        csv_path = model_dir / "effectiveness_best_practices.csv"

        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)

        response_model = MODEL_NAMES.get(model_dir.name, model_dir.name)

        for _, r in df.iterrows():

            mode = MODE_NAMES.get(r["mode"], r["mode"])

            rows.append({
                "Judge Model": judge,
                "Response Model": response_model,
                "Mode": mode,
                "Protocol Effectiveness": r["protocol_effectiveness"],
                "CBT Best Practices": r["cbt_best_practices"]
            })

    return pd.DataFrame(rows)


def main():

    dfs = []

    for judge in JUDGES:
        dfs.append(load_judge_data(judge))

    df = pd.concat(dfs, ignore_index=True)

    # enforce logical mode ordering
    df["Mode"] = pd.Categorical(df["Mode"], categories=MODE_ORDER, ordered=True)

    # sort rows
    df = df.sort_values(
        ["Judge Model", "Response Model", "Mode"]
    )

    # save
    df.to_csv(OUT_FILE, index=False)

    print("\nSaved:", OUT_FILE)
    print("\nPreview:")
    print(df)


if __name__ == "__main__":
    main()