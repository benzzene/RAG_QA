from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document

from RAG.config import AppConfig, RuntimeConfig
from RAG.indexing.build_index import build_index
from RAG.indexing.embeddings.embeddings_factory import create_embeddings
from RAG.indexing.vector_store.faiss_index import FaissFlatStore
from RAG.indexing.build_manifest import build_manifest, load_manifest, manifest_diff
from RAG.models.model_factory import make_model
from RAG.reranker.cross_encoder import CrossEncoderReranker
from RAG.retrival.bm25 import BM25
from RAG.retrival.dense import DenseSearch
from RAG.retrival.retriver import Retrieved, Retriever
from RAG.retrival.rrf import RRF


@dataclass(frozen=True)
class IndexAssetsState:
    index_exists: bool
    chunks_exists: bool
    manifest_exists: bool
    manifest_diff: list[str]

    @property
    def assets_missing(self) -> bool:
        return not self.index_exists or not self.chunks_exists

    @property
    def manifest_missing(self) -> bool:
        return not self.manifest_exists

    @property
    def sources_changed(self) -> bool:
        return bool(self.manifest_diff)

    @property
    def needs_rebuild(self) -> bool:
        return self.assets_missing or self.manifest_missing or self.sources_changed


def inspect_index_assets(cfg: AppConfig) -> IndexAssetsState:
    """
    Inspect index assets and compare the saved manifest with the current state.

    This function does not build or modify anything. It is intended for CLI/API
    flows that need to tell the user whether the index is missing or outdated.
    """
    index_path = Path(cfg.runtime.index_path)
    chunks_path = Path(cfg.runtime.chunks_path)
    manifest_path = Path(cfg.index.manifest_path)

    current_manifest = build_manifest(cfg.index)
    saved_manifest = load_manifest(manifest_path)

    return IndexAssetsState(
        index_exists=index_path.exists(),
        chunks_exists=chunks_path.exists(),
        manifest_exists=manifest_path.exists(),
        manifest_diff=manifest_diff(saved_manifest, current_manifest),
    )


def ensure_index_assets(cfg: AppConfig) -> None:
    """
    Ensure required runtime assets exist.

    This function only builds the index when required runtime files are missing
    and runtime.build_if_missing is enabled. It does not rebuild the index only
    because source documents changed; that decision should be handled by CLI/API.
    """
    state = inspect_index_assets(cfg)

    if not state.assets_missing:
        return

    if not cfg.runtime.build_if_missing:
        raise FileNotFoundError(
            "Missing runtime assets: "
            f"index_exists={state.index_exists}, "
            f"chunks_exists={state.chunks_exists}. "
            "Enable runtime.build_if_missing or build the index first."
        )

    build_index(cfg.index)


def load_chunks(path: Path) -> List[Document]:
    chunks: List[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            chunks.append(
                Document(
                    page_content=row["page_content"],
                    metadata=row["metadata"],
                )
            )
    return chunks


def make_runtime(cfg: RuntimeConfig):
    embeddings = create_embeddings(
        {
            "provider": cfg.embeddings.provider,
            "device": cfg.embeddings.device,
            "model": cfg.embeddings.model,
            "normalize": cfg.embeddings.normalize,
            "batch_size": cfg.embeddings.batch_size,
            "similarity_metric": cfg.embeddings.similarity_metric,
        }
    )
    store = FaissFlatStore.load(cfg.index_path)
    chunks = load_chunks(Path(cfg.chunks_path))

    bm25 = BM25(chunks)
    dense = DenseSearch(index=store, embedder=embeddings)
    rrf = RRF()

    retriever = Retriever(
        bm25=bm25,
        dense=dense,
        rrf=rrf,
        chunks=chunks,
        k_bm25=cfg.retriever.k_bm25,
        k_dense=cfg.retriever.k_dense,
        topn=cfg.retriever.topn,
    )

    reranker = CrossEncoderReranker(
        model_name=cfg.reranker.model_name,
        device=cfg.reranker.device,
        max_length=cfg.reranker.max_length,
    )

    qwen = make_model(
        {
            "provider": cfg.llm.provider,
            "model_id": cfg.llm.model_id,
            "device": cfg.llm.device,
            "use_4bit": cfg.llm.use_4bit,
            "system_prompt": cfg.llm.system_prompt,
            "max_context_chars": cfg.llm.max_context_chars,
            "max_new_tokens": cfg.llm.max_new_tokens,
            "temperature": cfg.llm.temperature,
            "top_p": cfg.llm.top_p,
            "do_sample": cfg.llm.do_sample,
            "repetition_penalty": cfg.llm.repetition_penalty,
        }
    )

    return retriever, reranker, qwen

def run_queries(queries: List[str], cfg: AppConfig) -> List[Dict[str, Any]]:
    """Deprecated, use RAGService"""
    ensure_index_assets(cfg)
    retriever, reranker, qwen = make_runtime(cfg.runtime)
    results: List[Dict[str, Any]] = []

    for q in queries:
        candidates: List[Retrieved] = retriever.retrieve(q)
        reranked: List[Retrieved] = reranker.rerank(
            q,
            candidates,
            k_final=cfg.runtime.reranker.k_final,
            batch_size=cfg.runtime.reranker.batch_size,
        )
        ctxs = [{"text": r.text, "meta": r.meta, "score": r.score} for r in reranked]
        out = qwen.answer_from_contexts(q, ctxs)
        results.append(out)

    return results
