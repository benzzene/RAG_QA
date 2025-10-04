from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from langchain.schema import Document

@dataclass
class Retrieved:
    """Represents a retrieved chunk with its associated metadata.

    Attributes:
        idx (int): Index of the retrived chunk.
        score (float): Retrieval score indicating relevance to the query.
        text (str): The actual text content of the retrieved chunk.
        meta (Dict[str, Any]): Metadata associated with the chunk.
    """
    idx: int
    score: float
    text: str
    meta: Dict[str, Any]

class RRF:
    """Implements Reciprocal Rank Fusion (RRF) for combining retrieval results.

    This class provides methods for fusing ranked retrieval outputs from
    multiple retrieval models and for gathering the corresponding candidate
    chunks.
    """
        
    @staticmethod
    def rrf_fuse(rankings: List[List[Tuple[int, float]]], k_rrf: int = 60, topn: int = 80) -> List[int]:
        """Fuses multiple ranked lists into a single ranking using RRF.

        Args:
            rankings (List[List[Tuple[int, float]]]): A list of ranked lists. 
                Each ranked list is represented as tuples of (chunks id, score).
            k_rrf (int, optional): RRF constant to dampen the effect of rank position.
                Defaults to 60.
            topn (int, optional): Maximum number of indices to return. Defaults to 80.

        Returns:
            List[int]: A list of chunk indices ranked by fused score.
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
        """Collects candidate chunks based on fused indices.

        This method maps the indices from the fused ranking back to the
        corresponding documents (chunks) and wraps them in `Retrieved` objects.

        Args:
            indices (List[int]): List of chunk indices from the fused ranking.
            chunks (List[Document]): List of chunks to retrieve from.

        Returns:
            List[Retrieved]: A list of `Retrieved` objects with placeholder scores
            (to be filled later during reranking).
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
