from RAG.retrival.bm25 import BM25
from RAG.retrival.dense import DenseSearch
from RAG.retrival.rrf import RRF
from typing import List, Dict, Any
from langchain.schema import Document
from dataclasses import dataclass

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

class Retriever:
    """Retriever that combines BM25 and dense search results using RRF fusion.

    This class orchestrates multiple retrieval strategies (BM25 and dense vector
    search) and merges their results through reciprocal rank fusion (RRF).

    Attributes:
        bm25 (BM25): Instance of BM25 retrieval model.
        dense (DenseSearch): Instance of dense vector search model.
        rrf (RRF): Reciprocal rank fusion instance for merging results.
        chunks (List[Document]): List of document chunks to retrieve from.
        k_bm25 (int): Number of top results to fetch from BM25. Defaults to 60.
        k_dense (int): Number of top results to fetch from dense search. Defaults to 60.
        topn (int): Number of final results to return after fusion. Defaults to 80.
    """

    def __init__(self, bm25: BM25, dense: DenseSearch, rrf: RRF, chunks: List[Document], k_bm25: int = 60, k_dense: int = 60, topn: int = 80): 
        """
        Args:
            bm25 (BM25): Instance of BM25 retrieval model.
            dense (DenseSearch): Instance of dense vector search model.
            rrf (RRF): Reciprocal rank fusion instance for merging results.
            chunks (List[Document]): List of document chunks to retrieve from.
            k_bm25 (int, optional): Number of BM25 results to fetch. Defaults to 60.
            k_dense (int, optional): Number of dense search results to fetch. Defaults to 60.
            topn (int, optional): Number of final results after fusion. Defaults to 80.
        """
        self.bm25 = bm25
        self.dense = dense
        self.rrf = rrf
        self.chunks = chunks
        self.k_bm25 = k_bm25
        self.k_dense = k_dense
        self.topn = topn

    def retrieve(self, query:str) -> List[Retrieved]:
        """Retrieves the most relevant chunks for a given query.

        This method performs retrieval using BM25 and dense search, then merges
        the results using reciprocal rank fusion (RRF). Finally, it gathers the
        top candidate chunks.

        Args:
            query (str): The input query string.

        Returns:
            List[Retrieved]: A list of retrieved chunks with their metadata.
        """
        bm25_hits = self.bm25.bm25_search(query, self.k_bm25)
        dense_hits = self.dense.dense_search(query, self.k_dense)
        rrf_fusion = self.rrf.rrf_fuse(rankings=[bm25_hits, dense_hits], topn=self.topn)
        retrived_hits = self.rrf.gather_candidates(indices=rrf_fusion, chunks=self.chunks)

        return retrived_hits