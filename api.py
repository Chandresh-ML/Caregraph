"""
FastAPI backend for CareGraph.

Wraps the exact same node/orchestrator logic used by demo.py and
eval/run_eval.py -- no business logic lives in this file, it's routing
and marshalling only. Run with:

    uvicorn api:app --reload

Then open http://localhost:8000 for the chat UI (served from web/).
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import tools
from src.state import new_state

app = FastAPI(title="CareGraph", version="0.1.0", description="Support agent built for The Talent Hack")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_runner():
    """Same auto-detection demo.py uses: real compiled StateGraph if
    langgraph is installed, the dependency-free orchestrator otherwise."""
    try:
        from src.graph import build_graph
        compiled = build_graph()
        return (lambda state: compiled.invoke(state)), "langgraph"
    except ImportError:
        from src import orchestrator
        return orchestrator.run, "fallback-orchestrator"


_RUN, RUNNER_NAME = _get_runner()


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    user_id: str
    query: str
    intent: Optional[str] = None
    confidence: float = 0.0
    escalated: bool = False
    ticket_id: Optional[str] = None
    response: str = ""
    tool_name: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None
    retrieved_context: List[Dict[str, Any]] = []
    trace: List[str] = []


@app.get("/api/health")
def health():
    return {"status": "ok", "runner": RUNNER_NAME}


@app.get("/api/users")
def list_users():
    return tools.list_demo_users()


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")
    if not req.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id cannot be empty")

    state = new_state(req.user_id, req.message)
    result = _RUN(state)

    return ChatResponse(
        user_id=req.user_id,
        query=req.message,
        intent=result.get("intent"),
        confidence=result.get("intent_confidence", 0.0),
        escalated=result.get("escalate", False),
        ticket_id=result.get("ticket_id"),
        response=result.get("response", ""),
        tool_name=result.get("tool_name"),
        tool_result=result.get("tool_result"),
        retrieved_context=result.get("retrieved_context", []),
        trace=result.get("trace", []),
    )


# Static frontend. Mounted last so /api/* routes above take priority.
_WEB_DIR = Path(__file__).resolve().parent / "web"
if _WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
