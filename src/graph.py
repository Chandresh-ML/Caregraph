"""
CareGraph's LangGraph wiring. This is the primary deliverable for the
challenge's "Agentic AI Systems" / "LangGraph" requirement.

Requires: pip install -r requirements.txt (langgraph, langchain-core).
Node logic itself lives in nodes.py so it stays testable independently of
this file -- see eval/run_eval.py and orchestrator.py.
"""

from __future__ import annotations
from langgraph.graph import StateGraph, START, END

from .state import CareGraphState
from .nodes import (
    classify_intent_node,
    retrieve_context_node,
    billing_node,
    network_node,
    streaming_node,
    plan_node,
    retention_node,
    escalate_node,
    generate_response_node,
    route_by_intent,
    route_after_specialist,
)

SPECIALISTS = ["billing", "network", "streaming", "plan", "retention"]


def build_graph():
    graph = StateGraph(CareGraphState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("billing", billing_node)
    graph.add_node("network", network_node)
    graph.add_node("streaming", streaming_node)
    graph.add_node("plan", plan_node)
    graph.add_node("retention", retention_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("generate_response", generate_response_node)

    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "retrieve_context")

    # After retrieval, branch to the right specialist (or straight to a
    # human if intent is unclear).
    graph.add_conditional_edges(
        "retrieve_context",
        route_by_intent,
        {**{s: s for s in SPECIALISTS}, "escalate": "escalate"},
    )

    # Every specialist can still escalate (e.g. a disputed charge or a
    # major outage) even once intent is correctly known.
    for specialist in SPECIALISTS:
        graph.add_conditional_edges(
            specialist,
            route_after_specialist,
            {"resolved": "generate_response", "escalate": "escalate"},
        )

    graph.add_edge("generate_response", END)
    graph.add_edge("escalate", END)

    return graph.compile()
