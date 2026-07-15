"""
Runs CareGraph end-to-end against five representative support scenarios,
including the two flagship World Cup 2026 surge scenarios from the project
brief: a duplicate PPV charge, and a degraded live-match stream.

Usage:
    python demo.py

Automatically uses the real compiled LangGraph StateGraph if `langgraph`
is installed, otherwise falls back to the zero-dependency orchestrator
running the identical node/routing functions.
"""

from __future__ import annotations
import sys

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from src.state import new_state

SCENARIOS = [
    {
        "label": "Duplicate PPV charge (World Cup Final) -> should escalate",
        "user_id": "U1001",
        "query": "I was charged twice for the World Cup final pay-per-view and I want my money back.",
    },
    {
        "label": "Match won't load during peak hours -> should self-resolve",
        "user_id": "U1003",
        "query": "The India vs Brazil match keeps buffering and won't load, what's going on?",
    },
    {
        "label": "4K plan upgrade -> should self-resolve",
        "user_id": "U1004",
        "query": "I want to upgrade my plan so I can watch matches in 4K.",
    },
    {
        "label": "Retention / cancellation intent -> should self-resolve",
        "user_id": "U1005",
        "query": "I'm thinking of cancelling and switching to another provider.",
    },
    {
        "label": "Vague / low-confidence query -> should escalate immediately",
        "user_id": "U1002",
        "query": "I don't know what's wrong honestly, nothing is working, can someone just call me back.",
    },
]


def _get_runner():
    try:
        from src.graph import build_graph
        compiled = build_graph()
        print("[demo] langgraph found -- running the real compiled StateGraph.\n")
        return lambda state: compiled.invoke(state)
    except ImportError as e:
        from src import orchestrator
        print(
            f"[demo] langgraph not installed ({e.__class__.__name__}) -- running the "
            f"zero-dependency fallback orchestrator on the exact same node functions.\n"
            f"        `pip install -r requirements.txt` to exercise the real StateGraph.\n"
        )
        return orchestrator.run


def main():
    run = _get_runner()
    for scenario in SCENARIOS:
        state = new_state(scenario["user_id"], scenario["query"])
        result = run(state)

        print("=" * 78)
        print(f"SCENARIO: {scenario['label']}")
        print(f"USER ({scenario['user_id']}): {scenario['query']}")
        print("-" * 78)
        print(f"intent          : {result.get('intent')}  (confidence={result.get('intent_confidence')})")
        print(f"escalated       : {result.get('escalate', False)}")
        if result.get("ticket_id"):
            print(f"ticket_id       : {result.get('ticket_id')}")
        print(f"CareGraph reply : {result.get('response')}")
        print()

    print("=" * 78)
    print(f"Ran {len(SCENARIOS)} scenarios.")


if __name__ == "__main__":
    main()
