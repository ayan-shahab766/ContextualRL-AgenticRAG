"""FastAPI application factory for the RL multi-agent RAG service."""
from __future__ import annotations

import json
import time
import uuid
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from contextual_rl_agentic_rag.config import settings
from contextual_rl_agentic_rag.deps import RagContext
from contextual_rl_agentic_rag.ingestion.processors import CSVProcessor, ImageProcessor, PDFProcessor
from contextual_rl_agentic_rag.logging_config import configure_logging, get_logger
from contextual_rl_agentic_rag.rl.orchestrator import RLOrchestrator
from contextual_rl_agentic_rag.serving.security import make_api_key_dependency, read_upload

logger = get_logger(__name__)

# Cap the text sent for LLM graph extraction to bound token/cost usage.
GRAPH_EXTRACT_CHARS = 5000


def _configure_cors(app: FastAPI) -> None:
    origins = settings.cors_origins_list
    if not origins:
        logger.info("CORS disabled (same-origin only). Set CORS_ALLOW_ORIGINS to enable.")
        return
    wildcard = "*" in origins
    if wildcard:
        # Never combine wildcard origins with credentials (browser-rejected + CSRF risk).
        logger.warning("CORS configured with wildcard origin; credentials disabled.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=not wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def create_app(orchestrator: Optional[RLOrchestrator] = None) -> FastAPI:
    configure_logging(settings.log_level)
    settings.ensure_dirs()

    if not settings.api_key:
        logger.warning("API_KEY not set: authentication is DISABLED. Set API_KEY in production.")

    orch = orchestrator or RLOrchestrator()
    ctx: RagContext = orch.ctx

    processors = {
        "pdf": PDFProcessor(settings),
        "image": ImageProcessor(ctx.llm),
        "csv": CSVProcessor(),
    }

    require_api_key = make_api_key_dependency(settings)
    app = FastAPI(
        title="ContextualRL-AgenticRAG: Contextual RL Multi-Agent Retrieval",
        version="4.0.0",
        description="Evidence-grounded, cost-aware multi-agent retrieval with contextual bandit routing.",
    )
    _configure_cors(app)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %s (%.1f ms)",
            request.method, request.url.path, response.status_code, elapsed_ms,
        )
        response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.1f}"
        return response

    @app.get("/")
    async def root():
        return {"message": "ContextualRL-AgenticRAG API v4.0", "capabilities": ["contextual_routing", "adaptive_retrieval", "evidence_grounding"]}

    @app.post("/upload", dependencies=[Depends(require_api_key)])
    async def upload_file(file: UploadFile = File(...), metadata: Optional[str] = Form(None)):
        data, safe_name, file_type = await read_upload(file, settings)

        file_id = str(uuid.uuid4())
        file_path = settings.upload_dir / f"{file_id}_{safe_name}"
        file_path.write_bytes(data)

        try:
            meta = json.loads(metadata) if metadata else {}
            if not isinstance(meta, dict):
                meta = {}
        except (json.JSONDecodeError, TypeError):
            meta = {}
        meta.update({"file_id": file_id, "source_filename": safe_name})

        try:
            result = processors[file_type].process(str(file_path), meta)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Processing failed for %s", safe_name)
            raise HTTPException(status_code=422, detail=f"Failed to process file: {exc}") from exc

        vs = ctx.vector_store
        counts: dict = {}
        counts["naive_rag"] = vs.add_documents("naive_rag", result["standard_chunks"])
        counts["sentence_window_rag"] = vs.add_documents(
            "sentence_window_rag", result["sentence_chunks"]
        )
        vs.add_documents("parent_child_rag_children", result["child_chunks"])
        counts["parent_child_rag"] = vs.add_documents(
            "parent_child_rag_parents", result["parent_chunks"]
        )
        counts["multimodal_rag"] = vs.add_documents("multimodal_rag", result["standard_chunks"])
        if file_type == "csv":
            counts["table_rag"] = vs.add_documents("table_rag", result["standard_chunks"])
        if ctx.bm25:
            ctx.bm25.add_documents(result["standard_chunks"])
            counts["bm25"] = len(result["standard_chunks"])
        if result["full_text"] and ctx.kg:
            ctx.kg.extract_and_add(result["full_text"][:GRAPH_EXTRACT_CHARS], source=safe_name)
            counts["graph_rag"] = ctx.kg.get_stats()

        return JSONResponse(
            content={
                "file_id": file_id,
                "filename": safe_name,
                "file_type": file_type,
                "index_counts": counts,
                "status": "indexed_to_all_agents",
            }
        )

    @app.post("/query", dependencies=[Depends(require_api_key)])
    async def query(question: str = Form(...), top_k: Optional[int] = Form(None)):
        return JSONResponse(content=await orch.query(question, top_k=top_k))

    @app.post("/query/{agent_name}", dependencies=[Depends(require_api_key)])
    async def query_agent(agent_name: str, question: str = Form(...), top_k: Optional[int] = Form(None)):
        return JSONResponse(content=await orch.query(question, agent_name=agent_name, top_k=top_k))

    @app.post("/compare", dependencies=[Depends(require_api_key)])
    async def compare(
        question: str = Form(...),
        agents: str = Form("naive_rag,agentic_rag,hybrid_rag"),
        top_k: Optional[int] = Form(None),
    ):
        agent_list = [a.strip() for a in agents.split(",") if a.strip()]
        return JSONResponse(content=await orch.compare(question, agent_list, top_k))

    @app.post("/feedback", dependencies=[Depends(require_api_key)])
    async def submit_feedback(
        interaction_id: str = Form(...),
        reward: float = Form(...),
        faithfulness: Optional[float] = Form(None),
        citation_quality: Optional[float] = Form(None),
        usefulness: Optional[float] = Form(None),
    ):
        if not 0.0 <= reward <= 1.0:
            raise HTTPException(status_code=400, detail="Reward must be between 0.0 and 1.0")
        dimensions = {"faithfulness": faithfulness, "citation_quality": citation_quality, "usefulness": usefulness}
        if any(value is not None and not 0.0 <= value <= 1.0 for value in dimensions.values()):
            raise HTTPException(status_code=400, detail="Feedback dimensions must be between 0.0 and 1.0")
        return JSONResponse(content=orch.submit_feedback(interaction_id, reward, **dimensions))

    @app.get("/rl/stats")
    async def rl_stats():
        return {
            "bandit_state": orch.bandit.get_stats(),
            "description": "Average reward per agent per question category. Higher = better.",
        }

    @app.get("/rl/leaderboard")
    async def rl_leaderboard():
        return {
            "leaderboard": orch.bandit.get_leaderboard(),
            "description": "Agents ranked by overall average reward across all categories.",
        }

    @app.get("/rl/contextual-stats")
    async def contextual_routing_stats():
        return {
            "contexts": orch.bandit.get_contextual_stats(),
            "description": "Per-context policy state keyed by category, domain, modality, and complexity.",
        }

    @app.post("/rl/optimize", dependencies=[Depends(require_api_key)])
    async def rl_optimize():
        return JSONResponse(content=orch.improver.analyze())

    @app.post("/rl/reset", dependencies=[Depends(require_api_key)])
    async def rl_reset():
        orch.bandit.reset()
        return {"status": "bandit_reset_to_initial_priors"}

    @app.get("/agents")
    async def list_agents():
        return orch.list_agents()

    @app.get("/collections")
    async def list_collections():
        kg_stats = ctx.kg.get_stats() if ctx.kg else {"nodes": 0, "edges": 0}
        return {
            "vector_stores": ctx.vector_store.list_collections(),
            "knowledge_graph": kg_stats,
        }

    @app.get("/health")
    async def health():
        return {
            "status": "running",
            "agents": len(orch.agents),
            "version": "4.0.0",
            "routing_policy": "contextual_ucb1",
            "llm_backend": settings.llm_backend,
            "vector_store": settings.vector_store,
            "auth_enabled": bool(settings.api_key),
        }

    return app


app = create_app()
