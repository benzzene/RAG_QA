from sentence_transformers import CrossEncoder
from RAG.retrival.retriver import Retrieved
from typing import List
import numpy as np

class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda", max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.reranker = CrossEncoder(model_name, device=device, max_length=max_length)
        
    def rerank(self, query: str, candidates: List[Retrieved], k_final: int = 7, batch_size: int = 32) -> List[Retrieved]:
        """
        Zwraca top-k_final kandydatów posortowanych wg oceny cross-encodera.
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
