from __future__ import annotations

from RAG.runtime_core import run_queries


if __name__ == "__main__":
    queries = [
        "Jakie modele LLaMa są dostępne?",
        "Kto stworzył PLLuM?",
        "Jaki model najlepiej działa na GPU z 24 GB VRAM?",
        "kiedy było lądowanie na Księżycu?",
    ]
    outputs = run_queries(queries)
    for o in outputs:
        print("\n=== ANSWER ===\n", o["answer"])
        print("\n=== SOURCES ===")
        for c in o["contexts"]:
            src = c["meta"].get("source") or c["meta"].get("source_path")
            print("-", src, "| chunk:", c["meta"].get("chunk_id"))
