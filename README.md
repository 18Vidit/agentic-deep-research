# Agentic Deep Research Pipeline

An autonomous, multi-agent Retrieval-Augmented Generation (RAG) system designed to deeply research, synthesize, and cite literature across a 700+ paper corpus of recent (2024-2026) arXiv publications on LLM agents. 

This repository contains the full data pipeline, the agentic reasoning loop (Planner, Reflector, Verifier), the evaluation scripts for component ablation, and an interactive Streamlit console.

## Architecture

The system moves beyond "single-shot" RAG by implementing a resilient, stateful loop:
1. **The Planner:** Decomposes complex user queries into targeted sub-searches.
2. **The Retriever:** Queries a persistent ChromaDB vector index built from chunked arXiv PDFs.
3. **The Reflector:** A self-critique loop that evaluates if the retrieved evidence is sufficient to answer the prompt without hallucination. It triggers re-searches if data is lacking.
4. **The Synthesizer:** Generates the final natural language response.
5. **The Citation Verifier:** A strict post-processor that enforces exact-match inline citations `[arxiv_id]` and scrubs unsupported claims.

## Quickstart: Reproducing the Pipeline (Grader Instructions)

Per the evaluation requirements, this system can be run from a fresh clone using a single command. The scripts are entirely idempotent and rate-limit resilient.

### 1. Environment Setup
Create a `.env` file in the root directory and add your Google Gemini API key:
```text
GEMINI_API_KEY=your_api_key_here
```

### 2. Execute the Pipeline
This command will install dependencies, scrape the arXiv corpus, build the vector index, run the agent across the 30 evaluation questions (including all ablation configurations), and format the final JSONL files.

**For Linux/macOS:**
```bash
chmod +x run.sh
./run.sh
```

**For Windows (PowerShell):**
```powershell
.\run.ps1
```

*Note: The final formatted predictions will be output to the `predictions/` directory.*

## Interactive Agent Console

This repository includes a professional UI to trace the agent's internal reasoning and test the architecture dynamically. 

```bash
streamlit run src/app.py
```
**Features:**
* **Live Ablations:** Toggle the Planner, Reflector, and Verifier on/off via the sidebar to see how they impact output quality.
* **Execution Trace:** A collapsible trace view that exposes the agent's internal dialogue, intermediate retrievals, and critique decisions.
* **Engine Switching:** Dynamically swap between `gemini-flash-lite-latest` (recommended for massive throughput) and Pro-tier models. Rate limits are gracefully handled.

## Repository Structure

```text
.
├── src/
│   ├── phase1_index.py       # arXiv scraper, PDF parser, and ChromaDB indexer
│   ├── phase2_agent.py       # Core Agent logic (Planner, Reflector, Verifier)
│   ├── phase3_eval.py        # Ablation study and evaluation loop (Resume-State enabled)
│   ├── format_submission.py  # JSONL formatter for strict grader requirements
│   └── app.py                # Streamlit interactive UI
├── eval/
│   └── questions.jsonl       # The 30 ground-truth prompts
├── predictions/              # (Generated) Final formatted JSONL predictions
├── run.sh                    # Linux single-command execution
├── run.ps1                   # Windows single-command execution
├── requirements.txt          # Frozen dependencies
└── README.md
```

## Engineering Trade-offs & Notes
* **Model Selection:** The default backend utilizes `gemini-flash-lite-latest`. While larger models offer deeper multi-step reasoning, the Lite architecture was deliberately chosen for its massive throughput capabilities, preventing HTTP 429 quota exhaustion during the heavy iterative loops required by the Reflector agent.
* **Idempotency:** The pipeline safely caches the ChromaDB index and evaluation states. If execution is interrupted, running the master script again will resume exactly where it left off without wasting API calls or bandwidth.