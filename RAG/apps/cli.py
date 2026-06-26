from __future__ import annotations

import argparse
from typing import Any, Dict

from RAG.config import load_config
from RAG.indexing.build_index import build_index
from RAG.runtime.runtime_core import inspect_index_assets
from RAG.runtime.rag_service import RAGService


def print_result(out: Dict[str, Any], show_sources: bool = True) -> None:
    print("\n=== ANSWER ===")
    print(out["answer"])

    if not show_sources:
        return

    print("\n=== SOURCES ===")
    for c in out.get("contexts", []):
        meta = c.get("meta", {})
        src = meta.get("source") or meta.get("source_path") or "unknown"
        chunk_id = meta.get("chunk_id")
        score = c.get("score")

        print(f"- {src} | chunk: {chunk_id} | score: {score}")


def confirm_rebuild() -> bool:
    answer = input("\nRebuild index now? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def check_index(cfg, *, force: bool, skip_check: bool) -> None:
    if force:
        print("Rebuilding index...")
        build_index(cfg.index)
        return

    if skip_check:
        return

    state = inspect_index_assets(cfg)

    if not state.needs_rebuild:
        return

    print("\nIndex requires attention:\n")

    if state.assets_missing:
        print("- index or chunks are missing")

    if state.manifest_missing:
        print("- manifest is missing")

    for reason in state.manifest_diff:
        print(f"- {reason}")

    if confirm_rebuild():
        print("\nRebuilding index...")
        build_index(cfg.index)
    else:
        print("\nUsing existing index.")


def interactive_mode(rag: RAGService, show_sources: bool) -> None:
    print("\n\n\n Welcome to RAG CLI.\n Ask your question \n Type 'exit' or 'quit' to close.\n")

    while True:
        query = input("> ").strip()

        if not query:
            continue

        if query.lower() in {"exit", "quit", "q"}:
            break

        out = rag.answer(query)
        print_result(out, show_sources=show_sources)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask questions using the RAG pipeline."
    )

    parser.add_argument(
        "query",
        nargs="*",
        help="Question to ask. If omitted, interactive mode starts.",
    )

    parser.add_argument(
        "--config",
        default="config/default_config.yaml",
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Do not print retrieved source chunks.",
    )

    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the index before starting.",
    )

    parser.add_argument(
        "--skip-index-check",
        action="store_true",
        help="Skip manifest/index freshness check.",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    check_index(
        cfg,
        force=args.rebuild_index,
        skip_check=args.skip_index_check,
    )

    rag = RAGService(cfg)
    show_sources = not args.no_sources

    if args.query:
        query = " ".join(args.query)
        out = rag.answer(query)
        print_result(out, show_sources=show_sources)
    else:
        interactive_mode(rag, show_sources=show_sources)


if __name__ == "__main__":
    main()