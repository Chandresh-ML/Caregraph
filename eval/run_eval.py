"""
Minimal evaluation harness for CareGraph's intent classifier and retriever.

Usage (from the project root):
    python eval/run_eval.py

This runs against whichever LLM backend get_llm_backend() selects --
MockLLM by default, or OpenAILLM if OPENAI_API_KEY is set and
langchain_openai is installed. Same eval set, same script, either
backend -- that's the point of keeping classify_intent() behind one
interface: routing quality can be tracked over time and compared across
backends/prompt versions, the same way you'd track eval metrics across
model or prompt revisions in an RLHF pipeline.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import get_llm_backend  # noqa: E402
from src.rag import FAQRetriever  # noqa: E402

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"


def run_classification_eval() -> float:
    llm = get_llm_backend()
    with open(EVAL_SET_PATH) as f:
        examples = json.load(f)

    rows = []
    correct = 0
    for ex in examples:
        pred_intent, confidence = llm.classify_intent(ex["query"])
        ok = pred_intent == ex["expected_intent"]
        correct += int(ok)
        rows.append((ex["query"], ex["expected_intent"], pred_intent, confidence, ok))

    print(f"Intent classification -- backend: {llm.name}")
    print("-" * 92)
    for query, expected, pred, conf, ok in rows:
        mark = "PASS" if ok else "FAIL"
        q_trunc = (query[:52] + "...") if len(query) > 55 else query
        print(f"[{mark}] expected={expected:<10} predicted={pred:<10} conf={conf:<5} | {q_trunc}")
    accuracy = correct / len(examples)
    print("-" * 92)
    print(f"Accuracy: {correct}/{len(examples)} = {accuracy:.0%}\n")
    return accuracy


def run_retrieval_smoke_test() -> float:
    """Sanity check: for a query clearly about one topic, does the
    top-ranked retrieved chunk come from the matching topic?"""
    retriever = FAQRetriever()
    checks = [
        ("duplicate charge refund", "billing"),
        ("match buffering during live event", "streaming"),
        ("upgrade to 4k", "plan"),
        ("cancel my subscription", "retention"),
        ("router keeps disconnecting", "network"),
    ]
    print("Retrieval smoke test (top-1 topic match)")
    print("-" * 92)
    correct = 0
    for query, expected_topic in checks:
        hits = retriever.retrieve(query, k=1)
        top_topic = hits[0]["topic"] if hits else None
        ok = top_topic == expected_topic
        correct += int(ok)
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] query={query!r:<38} expected={expected_topic:<10} got={top_topic}")
    accuracy = correct / len(checks)
    print("-" * 92)
    print(f"Retrieval top-1 accuracy: {correct}/{len(checks)} = {accuracy:.0%}\n")
    return accuracy


if __name__ == "__main__":
    run_classification_eval()
    run_retrieval_smoke_test()
