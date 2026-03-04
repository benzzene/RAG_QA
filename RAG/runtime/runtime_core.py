# RAG_QA/RAG/runtime_core.py
from __future__ import annotations

from typing import List, Dict, Any

from RAG.indexing.embeddings.embeddings_factory import create_embeddings
from RAG.indexing.vector_store.IndexStore_factory import create_indexstore

from RAG.io.loaders import DirectoryDocumentLoader, LoaderConfig
from RAG.io.splitters import RecursiveSplitter, SplitterConfig

from RAG.retrival.retriver import Retriever, Retrieved
from RAG.retrival.bm25 import BM25
from RAG.retrival.dense import DenseSearch
from RAG.retrival.rrf import RRF

from RAG.reranker.cross_encoder import CrossEncoderReranker
from RAG.models.model_factory import make_model


def load_runtime():
    embeddings = create_embeddings(
        {"provider": "hf", "device": "cuda", "model": "BAAI/bge-m3", "normalize": True}
    )
    store = create_indexstore({"provider": "faiss_flat", "device": "cpu"}, embedder=embeddings)
    store.load(path="data/index.faiss")

    cfg = LoaderConfig(root_dir="docs")
    loader = DirectoryDocumentLoader(cfg)
    raw_docs = loader.load()

    cfg_split = SplitterConfig(chunk_size=1000, chunk_overlap=200)
    splitter = RecursiveSplitter(cfg_split)
    chunks = splitter.split(raw_docs)

    return embeddings, store, chunks


def make_runtime():
    embeddings, store, chunks = load_runtime()

    bm25 = BM25(chunks)
    dense = DenseSearch(index=store, embedder=embeddings)
    rrf = RRF()

    retriever = Retriever(
        bm25=bm25,
        dense=dense,
        rrf=rrf,
        chunks=chunks,
        k_bm25=60,
        k_dense=60,
        topn=40,
    )

    reranker = CrossEncoderReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        device="cuda",
        max_length=512,
    )

    qwen = make_model(
        {
            "provider": "qwen",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "device": "cuda",
            "use_4bit": True,
            "system_prompt": (
                "Jesteś asystentem QA. Odpowiadasz WYŁĄCZNIE na podstawie przekazanego kontekstu."
                "Twoim zadaniem jest wydobyć informację z kontekstu, nawet jeżeli są rozproszone. "
                "Cytuj źródła, gdy to możliwe."
            ),
            "max_context_chars": 12000,
            "max_new_tokens": 512,
            "temperature": 0.2,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty": 1.1,
        }
    )

    return retriever, reranker, qwen


def run_queries(queries: List[str]) -> List[Dict[str, Any]]:
    retriever, reranker, qwen = make_runtime()
    results: List[Dict[str, Any]] = []

    for q in queries:
        candidates: List[Retrieved] = retriever.retrieve(q)
        reranked: List[Retrieved] = reranker.rerank(q, candidates, k_final=10, batch_size=32)
        ctxs = [{"text": r.text, "meta": r.meta, "score": r.score} for r in reranked]
        out = qwen.answer_from_contexts(q, ctxs)
        results.append(out)

    return results