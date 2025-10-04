from sentence_transformers import CrossEncoder
from RAG.retrival.retriver import Retrieved
from typing import List
import numpy as np

class CrossEncoderReranker:
    """Reranks retrieved chunks using a cross-encoder model.

    This class applies a cross-encoder (e.g., BAAI/bge-reranker-v2-m3) to 
    score query–chunk pairs and sorts the candidate chunks based on these scores.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda", max_length: int = 512):
        """
        Args:
            model_name (str, optional): Hugging Face model name or path. 
                Defaults to "BAAI/bge-reranker-v2-m3".
            device (str, optional): Device to run the model on (e.g., "cpu" or "cuda").
                Defaults to "cuda".
            max_length (int, optional): Maximum input sequence length for the model.
                Defaults to 512.
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.reranker = CrossEncoder(model_name, device=device, max_length=max_length)
        
    def rerank(self, query: str, candidates: List[Retrieved], k_final: int = 7, batch_size: int = 32) -> List[Retrieved]:
        """Reranks candidate chunks using the cross-encoder.

        Each query–chunk pair is scored by the cross-encoder. Candidates are
        then sorted by their scores, and the top `k_final` are returned.

        Args:
            query (str): Input query string.
            candidates (List[Retrieved]): Candidate chunks retrieved from earlier 
                retrieval steps (e.g., BM25, dense search, RRF).
            k_final (int, optional): Number of top candidates to return. Defaults to 7.
            batch_size (int, optional): Batch size for scoring. Defaults to 32.

        Returns:
            List[Retrieved]: Top-ranked candidate chunks with updated scores and 
            additional metadata (including rank position).
        """
        pairs = [(query, c.text) for c in candidates]
        scores = self.reranker.predict(pairs, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
        order = np.argsort(scores)[::-1][:k_final]

        ranked = []
        for rank, i in enumerate(order, 1):
            c = candidates[int(i)]
            ranked.append(Retrieved(
                idx=c.idx,
                score=float(scores[i]),
                text=c.text,
                meta={**c.meta, "rank": rank}
            ))
        return ranked
