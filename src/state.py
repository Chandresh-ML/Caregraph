"""
State schema for the CareGraph agent.

LangGraph passes a single state object between nodes. Each node receives the
current state and returns a partial dict of updates, which LangGraph merges
back into the running state. Using a TypedDict keeps this explicit and
type-checkable without pulling in a hard dependency on pydantic.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict


class CareGraphState(TypedDict, total=False):
    # --- input ---
    user_id: str
    user_query: str

    # --- routing ---
    intent: Optional[str]          # billing | network | streaming | plan | retention | unknown
    intent_confidence: float       # 0.0 - 1.0

    # --- retrieval ---
    retrieved_context: List[Dict[str, Any]]  # FAQ/policy chunks pulled from rag.py

    # --- tool execution ---
    tool_name: Optional[str]
    tool_result: Optional[Dict[str, Any]]

    # --- escalation ---
    escalate: bool
    escalation_reason: Optional[str]

    # --- output ---
    response: Optional[str]
    ticket_id: Optional[str]

    # --- trace, useful for debugging + eval, not shown to the end user ---
    trace: List[str]


def new_state(user_id: str, user_query: str) -> CareGraphState:
    """Construct a fresh state for a single incoming query."""
    return CareGraphState(
        user_id=user_id,
        user_query=user_query,
        intent=None,
        intent_confidence=0.0,
        retrieved_context=[],
        tool_name=None,
        tool_result=None,
        escalate=False,
        escalation_reason=None,
        response=None,
        ticket_id=None,
        trace=[],
    )


def log(state: CareGraphState, message: str) -> None:
    """Small helper so every node can leave a breadcrumb in state['trace']."""
    state.setdefault("trace", []).append(message)
