from RAG.retrival.bm25 import BM25
from RAG.retrival.dense import DenseSearch
from RAG.retrival.rrf import RRF
from typing import List, Dict, Any
from langchain.schema import Document
from dataclasses import dataclass

@dataclass
class Retrieved:
    idx: int
    score: float
    text: str
    meta: Dict[str, Any]

class Retriever:
    def __init__(self, bm25: BM25, dense: DenseSearch, rrf: RRF, chunks: List[Document], k_bm25: int = 60, k_dense: int = 60, topn: int = 80): 
        self.bm25 = bm25
        self.dense = dense
        self.rrf = rrf
        self.chunks = chunks
        self.k_bm25 = k_bm25
        self.k_dense = k_dense
        self.topn = topn

    def retrieve(self, query:str) -> List[Retrieved]:
        bm25_hits = self.bm25.bm25_search(query, self.k_bm25)
        dense_hits = self.dense.dense_search(query, self.k_dense)
        rrf_fusion = self.rrf.rrf_fuse(rankings=[bm25_hits, dense_hits], topn=self.topn)
        retrived_hits = self.rrf.gather_candidates(indices=rrf_fusion, chunks=self.chunks)

        return retrived_hits