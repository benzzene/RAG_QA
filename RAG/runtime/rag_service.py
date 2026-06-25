from __future__ import annotations

from typing import Any, Dict, List

from RAG.config import AppConfig
from RAG.runtime.runtime_core import ensure_index_assets, make_runtime
from RAG.retrival.retriver import Retrieved


class RAGService:
    def __init__(self, cfg: AppConfig):
        """Initializes the RAG service and loads all runtime components.

        Args:
            cfg (AppConfig): Application configuration.
           """
        self.cfg = cfg

        ensure_index_assets(cfg)

        self.retriever, self.reranker, self.llm = make_runtime(cfg.runtime)

    def answer(self, query: str) -> Dict[str, Any]:
        """Answers a single user query using the RAG pipeline.

        The method retrieves relevant documents, reranks them,
        builds the final context and generates an answer with the LLM.

        Args:
            query (str): User question.

        Returns:
            Dict[str, Any]: Generated answer together with supporting contexts
                and generation metadata.
        """
        candidates: List[Retrieved] = self.retriever.retrieve(query)

        reranked: List[Retrieved] = self.reranker.rerank(
            query,
            candidates,
            k_final=self.cfg.runtime.reranker.k_final,
            batch_size=self.cfg.runtime.reranker.batch_size,
        )

        contexts = [
            {"text": r.text, "meta": r.meta, "score": r.score}
            for r in reranked
        ]

        return self.llm.answer_from_contexts(query, contexts)

    def batch_ask(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Answers multiple queries sequentially using the same runtime.

        Args:
            queries (List[str]): User queries.

        Returns:
            List[Dict[str, Any]]: Answers for all queries.
        """
        return [self.answer(q) for q in queries]