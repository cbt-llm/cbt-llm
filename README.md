# cbt-llm
CMSC691: CBT Oriented LLMs

## Project Setup

```sh
git clone https://github.com/cbt-llm/cbt-llm.git
cd cbt-llm
cp .env.example .env
```

### Add PyMedTermino-0.3.2
 
1. Under ```external_libs``` add the PyMedTermino-0.3.2 folder

2. Update ```external_libs/PyMedTermino-0.3.2/setup.py``` with the following:
```sh
import os
SNOMEDCT_DIR = os.getenv("SNOMEDCT_DIR")
SNOMEDCT_CORE_FILE = os.getenv("SNOMEDCT_CORE_FILE")
```



### Add SNOMED CT Data

PyMedTermino requires access to the official SNOMED CT distribution, which is protected by a copyright license. 
So we will not be committing it to the repo. You can obtain the license and then upload the release files locally 
into the project's ```data/external/snomed``` directory.

Accordingly, add the relative path to your ```.env```

It should look like:

```sh
SNOMEDCT_DIR = "data/external/snomed/SnomedCT_RF2Release_INT_20160731"
SNOMEDCT_CORE_FILE = "data/external/snomed/SNOMEDCT_CORE_SUBSET_201611/SNOMEDCT_CORE_SUBSET_201611.txt"
```

These paths point to the directory and core file that PyMedTermino will use. Then continue as follows:

```sh
python3 -m venv .venv
source .venv/bin/activate
make install
```

If make install fails:
```sh
pip install -e .
pip install external_libs/PyMedTermino-0.3.2
pip install -r requirements.txt
```

## Graph Setup

Create a Neo4j Instance and add the uri, username, password to your ```.env```

### Load From PymedTermino
```python -m cbt_llm.pymed_loader```

### Load SNOMED Concepts/Relationships to neo4j Graph
```python -m cbt_llm.pymed_graph```

## Graph Retrieval

### Create node embeddings and store in the graph
```python -m cbt_llm.main```


To check if the embeddings have been created in the neo4j
Run this below command in neo4j desktop/browser

```MATCH (n:Concept) WHERE n.embedding RETURN n.code AS code, size(n.embedding) AS embedding_size LIMIT 5```

### Install the Graph Data Science Plugin on the Neo4j Desktop 
This is used for semantic search. 

### Retrieve the top_k embeddings

```sh
python -m cbt_llm.main
```

Select option `1` (Neo4j SNOMED retrieval).

Output written to:
```
src/output_files/neo4j_retrival_output/snomed_turn_results.csv
```

Columns: `Embedding, Turn, User Text, SNOMED Term, Code, Score, Relation Type, Relation Target Code, Relation Target Term`

---

## NLI Re-ranking

After SNOMED retrieval, a Natural Language Inference (NLI) re-ranking layer filters noisy top-K findings using `cross-encoder/nli-deberta-v3-small`.

### How it works

For each retrieved SNOMED finding, a hypothesis is constructed from the finding term and its `INTERPRETS` relations:

```
"This person has anxiety about resuming sexual relations (finding),
 which involves psychological function, emotion, cognitive functions."
```

The NLI model scores `(user_text, hypothesis)` pairs and assigns a decision:

| NLI Label     | Condition                         | Decision |
| ------------- | --------------------------------- | -------- |
| ENTAILMENT    | highest score                     | KEEP     |
| NEUTRAL       | highest score AND score ≥ 0.5     | KEEP     |
| CONTRADICTION | highest score                     | DROP     |

### Run

```sh
python -m cbt_llm.main
```

Select option `4` (NLI re-ranking).

Or run directly without loading the embedding models:

```sh
python -c "from cbt_llm.pipelines.nli_reranker import run_nli_reranker; run_nli_reranker()"
```

### Outputs

Both files are written to `src/output_files/neo4j_retrival_output/`.

**`nli_reranked_results.csv`** — full results with NLI scores and decision per finding:

```
Turn, User Text, SNOMED Term, Code, Retrieval Score, Hypothesis,
NLI Label, Entailment Score, Neutral Score, Contradiction Score, Decision
```

**`nli_findings.json`** — kept SNOMED concept names per turn, for use in prompt integration:

```json
{
  "1": ["Breakup of romance (finding)", "Anxiety about resuming sexual relations (finding)"],
  "2": ["Anger (finding)", "Loss of control of anger (finding)"]
}
```

---

## Set Up LLMs

### Ollama (for local open models)

This project uses **Ollama** to run open-source LLMs locally (e.g., Gemma, Mistral).

1. **Install Ollama**

Follow the official installation instructions for your OS:
[https://ollama.com/download](https://ollama.com/download)

Verify installation:

```sh
ollama --version
```

2. **Pull required models**

Pull the models used in the experiments:

```sh
ollama pull gemma:2b
ollama pull mistral:7b-instruct
```

3. **Start the Ollama server**

Ollama must be running before launching the UI or experiments:

```sh
ollama serve
```

> Leave this process running in a separate terminal.
### OpenAI (for closed models)

To run experiments with GPT-4o, set your OpenAI API key:

```sh
export OPENAI_API_KEY=your_api_key_here
```

## Start User Interface

1. **Start Ollama (required for local models)**

In a separate terminal:

```sh
ollama serve
```

2. **Launch the Streamlit app**

From the project root directory:

```sh
export PYTHONPATH=src
streamlit run app.py
```

Once running, open the provided local URL in your browser.

![Chat Interface](images/ui.png)
![Conversation](images/chat.png)

---

We evaluate CBT-aligned retrieval-augmented generation using a factorial design, varying:

* **Prompting mode**: baseline vs. CBT-guided
* **Language model**: three open and closed LLMs

This results in a **2 × 3 factorial design**.

### Models used

| Model               | Argument  | Backend    |
| ------------------- | --------- | ---------- |
| Gemma 2B            | `gemma`   | Ollama     |
| Mistral 7B Instruct | `mistral` | Ollama     |
| GPT-4o              | `gpt`     | OpenAI API |

Each model is run under both **baseline** and **CBT-guided** system prompts.
Outputs are written to:

```
output/{model}/
```

### Prerequisites

* Ollama server running (for Gemma and Mistral)
* OpenAI API key set (for GPT-4o)

```sh
export PYTHONPATH=src
export OPENAI_API_KEY=your_api_key_here
ollama serve
```

### Run experiments

1. Compute Retrieval Recall Scores:
Evaluate SNOMED CT concept retrieval performance across the following embedding models:

- MPNet (all-mpnet-base-v2)
- SapBERT (SapBERT-from-PubMedBERT-fulltext)
- BioRedditBERT
- MentalBERT

```sh
python3 src/evaluation/recall_eval.py
```

Files generated:

- `recall_summary.csv`: Recall@1, @3, @5 per embedding model
- `qualitative_examples.csv`: Per-query retrieved concepts with correctness labels

2. Run Concept generation pipelines:

- `llm_pipeline` — LLM-based semantic extraction (option `2`)
- `neo4j_pipeline` — SNOMED CT graph retrieval via MPNet embeddings (option `1`)

3. Run NLI Re-ranking:

Filter retrieved SNOMED findings using NLI (option `4`). Produces `nli_reranked_results.csv` and `nli_findings.json`.

4. Generate Transcripts

Run 

```sh
./run_experiments.sh [baseline|cbt] ${MODEL}
```

Example:

```sh
./run_experiments.sh cbt gemma
./run_experiments.sh baseline mistral
./run_experiments.sh cbt gpt
```

## Generate Evaluation Plots

#### Therapist-Side Evaluation

We evaluate therapist responses using an LLM-as-a-judge framework that scores:

- Validation & reflection
- Socratic questioning
- Cognitive restructuring
- Overall CBT quality

Outputs are saved to:

```
evaluation/{model}/summary.csv
evaluation/{model}/*.judge.jsonl
```

Commands:

```python src/cbt_llm/plot_cbt_eval.py --models gpt gemma mistral```

#### Patient-Side Sentiment Evaluation

Patient responses are analyzed using VADER compound sentiment scores as a proxy for emotional trajectory.

Generate Patient Evaluation Plots

This script produces:

Patient sentiment trajectories (baseline vs. CBT-guided)

Protocol-specific patient sentiment effects

Cross-model aggregated patient effect

```python src/cbt_llm/plot_patient_eval.py --models gpt gemma mistral```

Outputs are saved to:

```
evaluation/patient_plots/
```


