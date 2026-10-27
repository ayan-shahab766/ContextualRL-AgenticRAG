"""Offline evaluation / A-B testing harness for the multi-agent RAG system."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from contextual_rl_agentic_rag.llm.base import LLMBackend, LLMError
from contextual_rl_agentic_rag.logging_config import configure_logging, get_logger
from contextual_rl_agentic_rag.rl.orchestrator import RLOrchestrator

logger = get_logger(__name__)

EVAL_DIR = Path("data/eval")
EVAL_QUERIES_PATH = EVAL_DIR / "queries.jsonl"

_DEMO_QUERIES = [
    {"id": "demo_1", "question": "Give a concise summary of the main topics covered in the uploaded documents."},
    {"id": "demo_2", "question": "List three key risks mentioned in the documents and explain why they matter."},
]


def load_eval_queries(max_questions: int | None = None) -> list[dict[str, Any]]:
    if EVAL_QUERIES_PATH.exists():
        queries: list[dict[str, Any]] = []
        with open(EVAL_QUERIES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line))
    else:
        queries = list(_DEMO_QUERIES)
    return queries[:max_questions] if max_questions is not None else queries


def judge_with_llm(llm: LLMBackend, question: str, answers: dict[str, str]) -> dict[str, Any]:
    """Use the configured LLM as an impartial judge to score each agent's answer."""
    system = (
        "You are an impartial evaluator for a retrieval-augmented QA system. You will be given a "
        "user question and multiple agent answers. Score each answer between 0.0 (useless/wrong) and "
        "1.0 (fully correct, concise, grounded). Prefer specific, grounded, honest answers."
    )
    answers_block = "\n\n".join(f"[{agent}]\n{text}" for agent, text in answers.items())
    user = (
        f"Question:\n{question}\n\nAgent answers:\n{answers_block}\n\n"
        'Respond in JSON: {"scores": {"<agent>": 0.0-1.0, ...}, '
        '"winner": "<agent>", "explanation": "brief"}'
    )
    try:
        return llm.complete_json(system, user)
    except LLMError as exc:
        logger.warning("Judge failed, assigning neutral scores: %s", exc)
        return {
            "scores": {a: 0.5 for a in answers},
            "winner": sorted(answers)[0] if answers else "",
            "explanation": f"Fallback scoring: {exc}",
        }


async def run_eval(
    agents: list[str], top_k: int, max_questions: int | None, judge: str
) -> dict[str, Any]:
    orchestrator = RLOrchestrator()
    queries = load_eval_queries(max_questions)
    results: list[dict[str, Any]] = []

    for q in queries:
        question = q["question"]
        qid = q.get("id") or q.get("name") or question[:32]
        comps = (await orchestrator.compare(question, agents, top_k=top_k)).get("comparisons", {})

        answers: dict[str, str] = {}
        for name, payload in comps.items():
            if isinstance(payload, dict):
                text = (
                    payload.get("answer")
                    or payload.get("response")
                    or payload.get("error")
                    or json.dumps(payload)
                )
            else:
                text = str(payload)
            answers[name] = text

        record: dict[str, Any] = {"id": qid, "question": question, "answers": answers}
        if judge != "none":
            record["judgement"] = judge_with_llm(orchestrator.ctx.llm, question, answers)
        results.append(record)

    summary: dict[str, Any] = {
        "agents": agents,
        "top_k": top_k,
        "num_questions": len(results),
        "per_agent": {},
    }
    if judge != "none":
        per_agent_scores: dict[str, list[float]] = {a: [] for a in agents}
        win_counts: dict[str, int] = {a: 0 for a in agents}
        for r in results:
            j = r.get("judgement") or {}
            for a, s in (j.get("scores") or {}).items():
                if a in per_agent_scores:
                    per_agent_scores[a].append(float(s))
            winner = j.get("winner")
            if winner in win_counts:
                win_counts[winner] += 1
        for a in agents:
            s = per_agent_scores[a]
            summary["per_agent"][a] = {
                "avg_score": round(sum(s) / len(s), 3) if s else 0.0,
                "num_scored": len(s),
                "wins": win_counts.get(a, 0),
            }

    return {"summary": summary, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline eval / A-B testing for multi-agent RAG.")
    parser.add_argument("--agents", type=str, default="naive_rag,agentic_rag,hybrid_rag")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-questions", type=int, default=5)
    parser.add_argument("--judge", type=str, choices=["none", "llm"], default="none")
    parser.add_argument("--output", type=str, default="data/eval/results.json")
    args = parser.parse_args()

    configure_logging()
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    result = asyncio.run(
        run_eval(agents=agents, top_k=args.top_k, max_questions=args.max_questions, judge=args.judge)
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=== EVAL SUMMARY ===")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nFull results written to: {output_path}")


if __name__ == "__main__":
    main()
