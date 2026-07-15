"""
Retrieval-Augmented Generation over data/faq_corpus.json.

Design choice: retrieval uses scikit-learn TF-IDF + cosine similarity
instead of embedding-API calls. Two reasons:
  1. It keeps the retriever runnable offline, with zero API cost, which
     matters when you're iterating on routing/response quality hundreds
     of times a day during development.
  2. For a corpus this size (a few dozen policy snippets), lexical
     retrieval is competitive with embeddings and much cheaper to run.

Swapping this for an embedding-based store (FAISS/Chroma + OpenAI or a
local embedding model) is a one-file change -- see README "Next steps".
It's wrapped as a LangChain BaseRetriever when langchain_core is available,
so the swap doesn't touch any calling code in nodes.py.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "faq_corpus.json"


class FAQRetriever:
    def __init__(self, corpus_path: Path = _CORPUS_PATH):
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.docs: List[Dict[str, Any]] = json.load(f)

        corpus_texts = [f"{d['title']}. {d['text']}" for d in self.docs]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(corpus_texts)

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked[:k]:
            if scores[i] <= 0:
                continue
            results.append({**self.docs[i], "score": round(float(scores[i]), 3)})
        return results


# Optional LangChain-native wrapper. Import is conditional so this module
# (and everything that depends on it) still works in an environment where
# langchain_core isn't installed -- see README on graceful degradation.
try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks import CallbackManagerForRetrieverRun

    class LangChainFAQRetriever(BaseRetriever):
        """Adapts FAQRetriever to LangChain's BaseRetriever interface so it
        can be dropped into LangChain chains/agents as a standard retriever."""

        faq_retriever: FAQRetriever
        k: int = 3

        def _get_relevant_documents(
            self, query: str, *, run_manager: "CallbackManagerForRetrieverRun"
        ) -> List["Document"]:
            hits = self.faq_retriever.retrieve(query, k=self.k)
            return [
                Document(page_content=h["text"], metadata={"id": h["id"], "title": h["title"], "score": h["score"]})
                for h in hits
            ]

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LangChainFAQRetriever = None  # type: ignore
    LANGCHAIN_AVAILABLE = False
