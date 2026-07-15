"""
Zero-dependency fallback orchestrator.

Runs the exact same node functions and routing functions defined in
nodes.py, via a plain Python loop instead of a compiled LangGraph
StateGraph. This exists so CareGraph's logic can be executed and demoed
anywhere -- including offline, or before `pip install langgraph` has run
-- without pretending to reimplement LangGraph itself.

`src.graph.build_graph()` is the version intended for grading/production;
demo.py prefers it automatically and only falls back to this module if
langgraph isn't importable.
"""

from __future__ import annotations
from typing import Any, Dict

from . import nodes
from .state import CareGraphState

SPECIALIST_NODES = {
    "billing": nodes.billing_node,
    "network": nodes.network_node,
    "streaming": nodes.streaming_node,
    "plan": nodes.plan_node,
    "retention": nodes.retention_node,
}


def run(state: CareGraphState) -> CareGraphState:
    state.update(nodes.classify_intent_node(state))
    state.update(nodes.retrieve_context_node(state))

    branch = nodes.route_by_intent(state)
    if branch == "escalate":
        state.update(nodes.escalate_node(state))
        return state

    state.update(SPECIALIST_NODES[branch](state))

    if nodes.route_after_specialist(state) == "escalate":
        state.update(nodes.escalate_node(state))
        return state

    state.update(nodes.generate_response_node(state))
    return state
