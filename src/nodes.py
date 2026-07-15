"""
Node functions for the CareGraph graph.

Each node is a plain `def f(state: CareGraphState) -> dict` that returns a
*partial* update, matching LangGraph's merge convention. They import
nothing from langgraph itself, which is a deliberate choice: it means
every node can be unit tested (see eval/run_eval.py) with a plain function
call, no graph compilation required, and it keeps the routing/business
logic reusable by the zero-dependency fallback orchestrator in
orchestrator.py.
"""

from __future__ import annotations
from typing import Any, Dict

from . import tools
from .llm import get_llm_backend, CONFIDENCE_THRESHOLD
from .rag import FAQRetriever
from .state import CareGraphState, log

_retriever = FAQRetriever()
_llm = get_llm_backend()


def classify_intent_node(state: CareGraphState) -> Dict[str, Any]:
    intent, confidence = _llm.classify_intent(state["user_query"])
    log(state, f"[classify_intent] intent={intent} confidence={confidence} (backend={_llm.name})")
    return {"intent": intent, "intent_confidence": confidence}


def retrieve_context_node(state: CareGraphState) -> Dict[str, Any]:
    hits = _retriever.retrieve(state["user_query"], k=3)
    log(state, f"[retrieve_context] {len(hits)} chunk(s): {[h['id'] for h in hits]}")
    return {"retrieved_context": hits}


def route_by_intent(state: CareGraphState) -> str:
    """Conditional edge after retrieve_context. Low confidence or an
    unrecognized intent both go straight to a human."""
    if state["intent"] == "unknown" or state["intent_confidence"] < CONFIDENCE_THRESHOLD:
        return "escalate"
    return state["intent"]


def route_after_specialist(state: CareGraphState) -> str:
    """Conditional edge after any specialist node: some tool results (a
    disputed charge, a major outage) can't be auto-resolved even once the
    intent is known correctly."""
    return "escalate" if state.get("escalate") else "resolved"


def billing_node(state: CareGraphState) -> Dict[str, Any]:
    payment_status = tools.check_payment_status(state["user_id"])
    log(state, f"[billing] check_payment_status={payment_status}")
    if payment_status.get("is_duplicate"):
        return {
            "tool_name": "check_payment_status",
            "tool_result": payment_status,
            "escalate": True,
            "escalation_reason": (
                f"Disputed duplicate charge of \u20b9{payment_status.get('amount_inr')} "
                f"for {payment_status.get('item')} -- requires manual verification."
            ),
        }
    bill = tools.get_bill(state["user_id"])
    return {"tool_name": "get_bill", "tool_result": bill, "escalate": False}


def network_node(state: CareGraphState) -> Dict[str, Any]:
    profile = tools.get_user_profile(state["user_id"])
    result = tools.check_network_status(profile["region"])
    log(state, f"[network] check_network_status={result}")
    if result.get("is_major_outage"):
        return {
            "tool_name": "check_network_status",
            "tool_result": result,
            "escalate": True,
            "escalation_reason": f"Major outage detected in {profile['region']}.",
        }
    return {"tool_name": "check_network_status", "tool_result": result, "escalate": False}


def streaming_node(state: CareGraphState) -> Dict[str, Any]:
    result = tools.check_stream_status(state["user_id"], event="live match")
    log(state, f"[streaming] check_stream_status={result}")
    if result.get("is_major_outage"):
        return {
            "tool_name": "check_stream_status",
            "tool_result": result,
            "escalate": True,
            "escalation_reason": "Major streaming outage, not routine congestion.",
        }
    if result.get("status") == "degraded":
        credit = tools.apply_service_credit(state["user_id"], reason="stream degraded during live event")
        result = {**result, **credit}
    return {"tool_name": "check_stream_status", "tool_result": result, "escalate": False}


def plan_node(state: CareGraphState) -> Dict[str, Any]:
    result = tools.get_plan_options(state["user_id"])
    log(state, f"[plan] get_plan_options={result}")
    return {"tool_name": "get_plan_options", "tool_result": result, "escalate": False}


def retention_node(state: CareGraphState) -> Dict[str, Any]:
    result = tools.get_retention_offer(state["user_id"])
    log(state, f"[retention] get_retention_offer={result}")
    return {"tool_name": "get_retention_offer", "tool_result": result, "escalate": False}


def escalate_node(state: CareGraphState) -> Dict[str, Any]:
    reason = state.get("escalation_reason") or "Confidence below auto-resolution threshold."
    ticket = tools.raise_ticket(state["user_id"], category=state.get("intent") or "unknown", details=reason)
    log(state, f"[escalate] raised ticket {ticket['ticket_id']} -- {reason}")
    profile = tools.get_user_profile(state["user_id"])
    response = (
        f"Hi {profile['name']}, I want to make sure this is handled properly, so I'm connecting you "
        f"with a specialist. I've opened ticket {ticket['ticket_id']} ({reason}) and included everything "
        f"you've told me so you won't need to repeat yourself."
    )
    return {"escalate": True, "ticket_id": ticket["ticket_id"], "response": response}


def generate_response_node(state: CareGraphState) -> Dict[str, Any]:
    profile = tools.get_user_profile(state["user_id"])
    ctx = {
        "user_name": profile["name"],
        "query": state["user_query"],
        "intent": state["intent"],
        "tool_result": state.get("tool_result"),
        "retrieved_context": state.get("retrieved_context"),
    }
    response = _llm.generate_response(ctx)
    log(state, "[generate_response] drafted final reply")
    return {"response": response}
