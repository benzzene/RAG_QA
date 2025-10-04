import numpy as np
import re
import unicodedata
from rank_bm25 import BM25Okapi
from typing import List, Any, Tuple
from langchain.schema import Document

_token_re = re.compile(r"\w+", flags=re.UNICODE)

def tokenize(text: str):
    """
    Normalize and tokenize input text.

    The function applies Unicode normalization (NFKC) and splits text into
    lowercase word tokens using a regex.

    Args:
        text (str): Input text.

    Returns:
        List[str]: List of normalized lowercase tokens.
    """
    text = unicodedata.normalize("NFKC", text)
    return [t.lower() for t in _token_re.findall(text)]


class BM25:
    """
    BM25 retriever for ranking text chunks.

    This class wraps the BM25Okapi implementation and provides
    document ranking based on tokenized text chunks.
    """

    def __init__(self, chunks: List[Document]):
        """
        Initialize BM25 retriever with pre-tokenized corpus.

        Args:
            chunks (List[Document]): List of LangChain Document objects
                representing text chunks to index.
        """
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(ch.page_content) for ch in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def bm25_search(self, query: str, k: int = 60) -> List[Tuple[int, float]]:
        """
        Perform BM25 ranking for the given query.

        Args:
            query (str): Query text.
            k (int, optional): Number of top results to return. Defaults to 60.

        Returns:
            List[Tuple[int, float]]: List of (chunk_index, score) tuples,
            sorted in descending order by BM25 score. Only results with
            scores > 0 are included.
        """
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0.0]
