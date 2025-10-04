from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from langchain.schema import Document

@dataclass
class Retrieved:
    idx: int
    score: float
    text: str
    meta: Dict[str, Any]

class RRF:
    @staticmethod
    def rrf_fuse(rankings: List[List[Tuple[int, float]]], k_rrf: int = 60, topn: int = 80) -> List[int]:
        """
        rankings: lista list wyników, każdy wynik to [(idx, score), ...] posortowany malejąco.
        Zwraca posortowaną listę indeksów chunków po fuzji RRF.

        k_rrf: uwaga!, k_rrf to stała używana w algorytmie.
        topn: ilość zwracanych topn kandydatów
        """
        fused = defaultdict(float)
        for rlist in rankings:
            rlist_sorted = sorted(rlist, key=lambda x: x[1], reverse=True)
            for rank, (idx, _score) in enumerate(rlist_sorted, start=1):
                fused[idx] += 1.0 / (k_rrf + rank)
        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _f in ordered[:topn]]

    @staticmethod
    def gather_candidates(indices: List[int], chunks: List[Document]) -> List[Retrieved]:
        """
        Zwraca kandydatów jako listę Retrieved na bazie indeksów chunków.
        """
        out: List[Retrieved] = []
        for i in indices:
            ch = chunks[i]
            out.append(Retrieved(
                idx=i,
                score=0.0,  # realny score pojawi się po rerankingu
                text=ch.page_content,
                meta={"source": ch.metadata.get("source_path"),
                    "chunk_id": ch.metadata.get("chunk_id", i)}
            ))
        return out
