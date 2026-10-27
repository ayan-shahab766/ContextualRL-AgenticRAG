## ContextualRL-AgenticRAG — Evidence-Grounded Multi-Agent Retrieval with Contextual Bandit Routing

ContextualRL-AgenticRAG is an end-to-end, production-style Retrieval-Augmented Generation platform that uses contextual reinforcement-learning routing to select, evaluate, and improve specialized RAG agents. Its policy layer optimizes answer quality, evidence quality, latency, and estimated serving cost.

- **10+ specialized RAG agents** (naive, sentence-window, parent-child, agentic, multimodal, table, graph, hybrid, HyDE, corrective).
- **Contextual UCB1 router** that learns which agent works best for a query's intent, domain, modality, and complexity—not just its broad question type.
- **Adaptive retrieval controller** that chooses a bounded retrieval depth and marks requests that need reranking or corrective fallback.
- **Evidence-grounded response contract**: every response includes stable evidence IDs, source locators, excerpts, and a transparent lexical grounding diagnostic.
- **Quality/cost/latency-aware feedback loop**: usefulness, faithfulness, and citation quality produce a routing reward adjusted for observed latency and estimated agent cost.
- **Hybrid retrieval stack**: dense vector search (Chroma) + BM25 + knowledge graph.
- **FastAPI backend** with clean, documented endpoints and Docker deployment.
 - **Offline eval + A/B testing**: `evals.py` runs multi-agent comparisons with optional LLM-as-judge scoring.
 - **Model adaptation pipeline**: documented loop for LoRA/SFT, distillation, quantization, and RL-based routing.

### ContextualRL-AgenticRAG policy flow

```
query → deterministic query signals → adaptive retrieval plan
      → contextual UCB1 agent selection → specialized RAG agent
      → evidence pack + grounding diagnostic → multi-signal feedback
      → quality/cost/latency-adjusted routing reward
```

The routing context is `category | domain | modality | complexity`. New contexts are warm-started from category priors, so the system retains sensible behavior before user feedback accumulates.
---

### High-Level Architecture

- **Ingestion layer**
  - `PDFProcessor`, `ImageProcessor`, `CSVProcessor` convert raw files into `Document` objects with rich metadata.
  - Multiple chunking strategies: standard chunks, sentence-level windows, parent/child hierarchy, table schema/row groups.
  - Indexes into:
    - Multiple **Chroma** collections (per RAG agent type),
    - **BM25** store for sparse retrieval,
    - **Knowledge graph** built from extracted entities/relations.

- **Agent layer (`agents/*.py`)**
  - Each agent encapsulates a retrieval strategy + prompting style.
  - Examples:
    - **SentenceWindowRAG**: retrieves sentence-level matches and expands with surrounding context.
    - **ParentChildRAG**: retrieves fine-grained child chunks but returns parent-level passages.
    - **GraphRAG**: traverses entities and relationships from the knowledge graph.
    - **MultimodalRAG**: indexes images via vision model descriptions.
    - **CorrectiveRAG**: detects when local retrieval is weak and falls back to external search.

- **RL Orchestrator (`contextual_rl_agentic_rag/rl/orchestrator.py`)**
  - Wraps all agents and exposes:
    - `query`: route a question to the best agent (or a user-selected agent).
    - `compare`: run multiple agents in parallel on the same question.
    - `submit_feedback`: update the bandit from user rewards.
  - Uses:
    - `BanditRouter` (UCB1 multi-armed bandit) for agent selection.
    - `classify_question` for cheap, non-LLM categorization (factual/analytical/relational/procedural/vague/visual).
    - `reward_store` to log all interactions and feedback.
    - `SelfImprover` to analyze historical data and recommend changes.

- **RL Bandit Router (`bandit_router.py`)**
  - Implements UCB1:
    - State: question **category**.
    - Actions: choose a **RAG agent**.
    - Reward: user feedback \(0.0 – 1.0\).
  - Uses **smart priors** so the system behaves well even before much data:
    - e.g. table queries initially favor `table_rag`, visual queries favor `multimodal_rag`, etc.
  - Persists state to `bandit_state.json` for continuity across restarts.

- **Feedback + Self-Improvement**
  - All interactions and rewards stream into `reward_log.jsonl`.
  - `SelfImprover` runs offline analysis:
    - Finds struggling vs. high-performing agents per category.
    - Learns which `top_k` values correlate with better rewards.
    - Detects hardest question categories and query-length effects.
    - Produces human-readable recommendations and parameter suggestions.

- **API Layer (`main.py`)**
  - `POST /upload` — upload PDFs/CSVs/images; auto-routed to correct processor; indexes into all relevant stores.
  - `POST /query` — ask a question; RL bandit chooses the best agent.
  - `POST /query/{agent}` — force a specific agent.
  - `POST /compare` — run the same question on multiple agents and compare outputs.
  - `POST /feedback` — send reward feedback (0.0–1.0) tied to an `interaction_id`.
  - `GET /rl/stats` — inspect bandit state (per-category performance).
  - `GET /rl/leaderboard` — global agent leaderboard.
  - `POST /rl/optimize` — run self-improvement analysis.
  - `GET /agents`, `GET /collections`, `GET /health` — observability and introspection.

For a deeper dive, see:
- `docs/ARCHITECTURE.md` — system architecture.
- `docs/MODEL_ADAPTATION.md` — LoRA/SFT, distillation, quantization, and eval loop.

---

### Project layout

```
contextual_rl_agentic_rag/
  config.py  schema.py  utils.py  logging_config.py
  llm/         # LLMBackend/EmbeddingBackend protocols + openai|local|fake + factory
  ingestion/   # PDF/Image/CSV processors
  retrieval/   # vector store (Chroma + in-memory), BM25, knowledge graph
  agents/      # 11 RAG agents (DI via RagContext)
  rl/          # bandit router, classifier, reward store, self-improver, orchestrator
  serving/     # FastAPI app factory, API-key auth, upload validation
  eval/        # offline eval / A-B harness
main.py evals.py     # API and eval-harness entrypoints
tests/               # test suite, run with only pytest
```

### Pluggable backends

The LLM and embeddings are selected via environment variables (see `.env.example`):

- `LLM_BACKEND` / `EMBEDDING_BACKEND`: `openai` (default) | `local` (sentence-transformers +
  extractive generation, no key) | `fake` (deterministic stub for tests/smoke runs).
- `VECTOR_STORE`: `chroma` (persistent) | `memory` (in-process, no native deps).

This means the system can run fully offline with **no API key** and is unit-testable without OpenAI,
Chroma, or a GPU.

---

### Running Locally (Dev)

- **1. Create and activate a virtualenv** (Python 3.10-3.11 recommended for the full stack)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# .\venv\Scripts\activate       # Windows
```

- **2. Install dependencies**

```bash
pip install -r requirements.txt
```

- **3. Configure environment**

```bash
cp .env.example .env            # then edit
# For a real run: set OPENAI_API_KEY and (recommended) API_KEY
# For an offline run: LLM_BACKEND=fake EMBEDDING_BACKEND=fake VECTOR_STORE=memory
```

- **4. Run the API**

```bash
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the Swagger UI. When `API_KEY` is set, sensitive endpoints
require the `X-API-Key` header.

### Offline run (no API key)

Set the fake/in-memory backends and start the app (or run the eval harness) with no external calls:

```bash
LLM_BACKEND=fake EMBEDDING_BACKEND=fake VECTOR_STORE=memory uvicorn main:app --port 8000
```

### Tests

```bash
pip install pytest
pytest        # 50 tests, no API key / Chroma / GPU required
```

---

### Example Workflow

- **Upload documents**

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/path/to/your.pdf"
```

- **Ask a question (bandit auto-routing)**

```bash
curl -X POST "http://localhost:8000/query" \
  -F "question=Summarize the key risks in this document." \
  -F "top_k=5"
```

- **Give feedback**

Take the `interaction_id` from the previous response and:

```bash
curl -X POST "http://localhost:8000/feedback" \
  -F "interaction_id=abcd1234" \
  -F "reward=0.9"
```

- **Inspect RL learning**

```bash
curl "http://localhost:8000/rl/stats"
curl "http://localhost:8000/rl/leaderboard"
```

---
