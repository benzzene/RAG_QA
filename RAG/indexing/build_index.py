from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from RAG.config import IndexConfig
from RAG.io.loaders import DirectoryDocumentLoader, LoaderConfig
from RAG.io.splitters import RecursiveSplitter, SplitterConfig
from RAG.indexing.embeddings.embeddings_factory import create_embeddings
from RAG.indexing.vector_store.IndexStore_factory import create_indexstore


"""
TODO:
Replace build_index() with an IndexBuilder class responsible for tracking
document changes and rebuilding the index when source documents are updated.
"""
def build_index(cfg: IndexConfig) -> None:
    """
    Build IndexStore from source documents.

    Args:
        cfg: Typed configuration for index building.
    """
    loader = DirectoryDocumentLoader(
        LoaderConfig(root_dir=cfg.docs_dir, show_progress=cfg.show_progress)
    )
    raw_docs = loader.load()

    splitter = RecursiveSplitter(
        SplitterConfig(
            chunk_size=cfg.splitter.chunk_size,
            chunk_overlap=cfg.splitter.chunk_overlap,
        )
    )
    chunks = splitter.split(raw_docs)

    embedder = create_embeddings(
        {
            "provider": cfg.embeddings.provider,
            "model": cfg.embeddings.model,
            "device": cfg.embeddings.device,
            "normalize": cfg.embeddings.normalize,
            "batch_size": cfg.embeddings.batch_size,
            "similarity_metric": cfg.embeddings.similarity_metric,
        }
    )

    texts = [chunk.page_content for chunk in chunks]
    vectors = embedder.embed_documents(texts).astype(np.float32)

    ids = np.array([int(chunk.metadata["chunk_id"]) for chunk in chunks], dtype=np.int64)

    store = create_indexstore(
        {
            "provider": cfg.store.provider,
            "device": cfg.store.device,
            "gpu_id": cfg.store.gpu_id,
        },
        embedder=embedder,
    )
    store.add_with_ids(vectors, ids)

    Path(cfg.index_path).parent.mkdir(parents=True, exist_ok=True)

    store.save(cfg.index_path)

    with open(cfg.chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            row = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved index to: {cfg.index_path}")
    print(f"Saved chunks to: {cfg.chunks_path}")
    print(f"Indexed chunks: {len(chunks)}")
