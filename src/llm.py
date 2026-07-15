"""
LLM backend with graceful degradation.

get_llm_backend() returns:
  - OpenAILLM  if OPENAI_API_KEY is set AND langchain_openai is installed
  - MockLLM    otherwise (deterministic, zero-cost, zero-dependency)

Why this matters beyond convenience: it means CareGraph's control flow,
routing logic, and escalation rules can be fully unit-tested and demoed
without spending API credits or needing network access -- the same
reasoning behind bringing per-user token costs down on a production LLM
product. The two backends implement the same two-method interface, so
nodes.py never needs to know which one it's talking to.
"""

from __future__ import annotations
import os
import re
from typing import Any, Dict, List, Tuple

CONFIDENCE_THRESHOLD = 0.55

_KEYWORDS: Dict[str, List[str]] = {
    "billing": ["charge", "charged", "bill", "billed", "refund", "money back", "twice", "duplicate", "invoice", "payment"],
    "network": ["wifi", "router", "internet connection", "no signal", "network is down", "outage"],
    "streaming": ["buffering", "won't load", "wont load", "not loading", "stream", "streaming", "match", "keeps freezing"],
    "plan": ["upgrade", "plan", "subscription", "4k", "package", "tier"],
    "retention": ["cancel", "cancelling", "cancellation", "switch", "leave the service", "unsubscribe", "close my account", "another provider"],
}
# Tiebreak order when two intents score equally on keyword hits.
_PRIORITY = ["billing", "network", "streaming", "plan", "retention"]


class MockLLM:
    """Deterministic, keyword-based stand-in for an LLM call. Used for
    offline development, unit tests, and CI -- not meant to replace real
    intent understanding in production."""

    name = "mock"

    def classify_intent(self, query: str) -> Tuple[str, float]:
        q = query.lower()
        scores = {intent: sum(1 for kw in kws if kw in q) for intent, kws in _KEYWORDS.items()}
        best_score = max(scores.values())
        if best_score == 0:
            return "unknown", 0.15
        # tie-break by fixed priority order
        best_intent = next(i for i in _PRIORITY if scores[i] == best_score)
        confidence = min(0.95, 0.40 + 0.18 * best_score)
        return best_intent, round(confidence, 2)

    def generate_response(self, ctx: Dict[str, Any]) -> str:
        name = ctx.get("user_name", "there")
        intent = ctx["intent"]
        tool_result = ctx.get("tool_result") or {}
        top_faq = (ctx.get("retrieved_context") or [{}])[0]

        if intent == "streaming":
            eta = tool_result.get("eta_minutes", "a few")
            return (
                f"Hi {name}, thanks for flagging that. Your stream is affected by "
                f"{tool_result.get('reason', 'a service issue')} -- this is on our side, not your "
                f"connection. It should clear up in about {eta} minutes. In the meantime, try switching "
                f"quality to Auto or reconnecting to the nearest server. We've applied a service credit "
                f"to your account for the disruption, no action needed from you."
            )
        if intent == "plan":
            opts = tool_result.get("options", [])
            best = next((o for o in opts if o.get("supports_4k")), opts[0] if opts else {})
            return (
                f"Hi {name}, your current plan is {tool_result.get('current_plan', 'Standard')}. "
                f"To stream in 4K you'll want {best.get('name', 'Premium')} at ₹{best.get('price_inr', '—')}/mo -- "
                f"it activates immediately and this cycle is pro-rated. Want me to switch you over now?"
            )
        if intent == "retention":
            return (
                f"Hi {name}, sorry to hear you're considering leaving. Before you go: we can offer "
                f"{tool_result.get('offer', 'a discount')}, or {tool_result.get('alternative', 'a free upgrade')}. "
                f"Either can be applied right now if you'd like to stay."
            )
        # generic fallback grounded in the top retrieved FAQ
        snippet = top_faq.get("text", "")
        return f"Hi {name}, here's what I found: {snippet}"


class OpenAILLM:
    """Production backend using LangChain's ChatOpenAI. Only imported/
    instantiated when OPENAI_API_KEY is present, so this file has zero
    hard dependency on langchain_openai being installed."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        self._chat = ChatOpenAI(model=model, temperature=temperature)
        self._classify_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Classify the customer support query into exactly one of: "
             "billing, network, streaming, plan, retention, unknown. "
             "Respond with just: <intent>,<confidence 0-1>"),
            ("human", "{query}"),
        ])
        self._response_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are CareGraph, a telecom support agent for Deutsche Telekom Digital Labs. "
             "Answer using ONLY the provided tool result and policy context. Be concise, "
             "warm, and specific. Never invent numbers that aren't in the context."),
            ("human",
             "Customer ({user_name}) asked: {query}\n\n"
             "Intent: {intent}\nTool result: {tool_result}\nPolicy context: {retrieved_context}"),
        ])

    def classify_intent(self, query: str) -> Tuple[str, float]:
        chain = self._classify_prompt | self._chat
        raw = chain.invoke({"query": query}).content.strip()
        try:
            intent, conf = raw.split(",")
            return intent.strip(), float(conf.strip())
        except Exception:
            return "unknown", 0.2

    def generate_response(self, ctx: Dict[str, Any]) -> str:
        chain = self._response_prompt | self._chat
        result = chain.invoke({
            "user_name": ctx.get("user_name", "there"),
            "query": ctx.get("query", ""),
            "intent": ctx.get("intent", ""),
            "tool_result": ctx.get("tool_result", {}),
            "retrieved_context": ctx.get("retrieved_context", []),
        })
        return result.content


def get_llm_backend():
    """Factory: real backend if we can build one, mock otherwise."""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAILLM()
        except ImportError:
            print("[llm] OPENAI_API_KEY is set but langchain_openai isn't installed -- "
                  "falling back to MockLLM. Run `pip install -r requirements.txt`.")
    return MockLLM()
