from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import json
import sys
import traceback

from RAG.io.loaders import LoaderConfig, DirectoryDocumentLoader
from RAG.io.splitters import SplitterConfig, RecursiveSplitter
from RAG.indexing.embeddings.embeddings_factory import create_embeddings
from RAG.indexing.vector_store.IndexStore_factory import create_indexstore
from RAG.indexing.vector_store.builder import IndexBuilder


def build_pipeline(
    *,
    docs_dir: str = "docs",
    out_dir: str = "data",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embed_cfg: Mapping[str, Any] | None = None,
    store_cfg: Mapping[str, Any] | None = None,
    instr_prefix: str = "",
) -> None:
    """
    0. I/O ścieżki
    1. load → split (DirectoryDocumentLoader + RecursiveSplitter)
    2. fabryki: embeddings + vector store
    3. builder: build -> save artefakty (faiss, ids, metadata, manifest)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.faiss"
    ids_path = out / "ids.npy"
    metadata_path = out / "metadata.jsonl"
    manifest_path = out / "manifest.json"

    print(f"[I/O] Wyjście: {out.resolve()}")

    print(f"[LOAD] Wczytywanie dokumentów z: {docs_dir}")
    loader = DirectoryDocumentLoader(LoaderConfig(root_dir=docs_dir))
    raw_docs = loader.load()
    print(f"[LOAD] Wczytano: {len(raw_docs)} dokumentów")

    print(f"[SPLIT] chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    splitter = RecursiveSplitter(SplitterConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    chunks = splitter.split(raw_docs)
    print(f"[SPLIT] Utworzono: {len(chunks)} chunków")

    embed_cfg = embed_cfg or {"provider": "hf", "device": "cuda"}
    print(f"[EMB] Tworzenie embeddera: {embed_cfg}")
    embeddings = create_embeddings(embed_cfg)

    store_cfg = store_cfg or {"provider": "faiss_flat", "device": "cpu"}
    print(f"[STORE] Tworzenie vector store: {store_cfg}")
    vector_store = create_indexstore(store_cfg, embedder=embeddings)

    print("[BUILD] Budowanie indeksu…")
    builder = IndexBuilder(embeddings, vector_store)
    result = builder.build_from_chunks(chunks=chunks, instr_prefix=instr_prefix)
    print("[BUILD] Zbudowano indeks")

    print("[SAVE] Zapisywanie artefaktów…")
    builder.save_artifacts(
        result,
        index_path=str(index_path),
        ids_path=str(ids_path),
        metadata_jsonl_path=str(metadata_path),
        manifest_path=str(manifest_path),
        extras={
            "pipeline": "v1",
            "docs_dir": docs_dir,
            "embed_cfg": dict(embed_cfg),
            "store_cfg": dict(store_cfg),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
    )
    print(f"[DONE] Zapisano:\n - {index_path}\n - {ids_path}\n - {metadata_path}\n - {manifest_path}")


def main() -> None:
    try:
        build_pipeline(
            docs_dir="docs",
            out_dir="data",
            chunk_size=1000,
            chunk_overlap=200,
            embed_cfg={"provider": "hf", "device": "cuda", "model": "BAAI/bge-m3", "normalize": True},
            store_cfg={"provider": "faiss_flat", "device": "cpu"},  
            instr_prefix="", 
        )
    except Exception as e:
        print("[ERROR] Pipeline przerwane:", e, file=sys.stderr)
        traceback.print_exc()
        try:
            Path("data").mkdir(parents=True, exist_ok=True)
            (Path("data") / "pipeline_error.json").write_text(
                json.dumps({"error": str(e)}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
