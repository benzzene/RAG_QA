from __future__ import annotations

import argparse
from typing import Any, Dict

from RAG.config import load_config
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

    args = parser.parse_args()

    cfg = load_config(args.config)
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