from typing import List, Tuple
from RAG.indexing.embeddings.embeddings import Embeddings
from RAG.indexing.embeddings.hf_embeddings import HFEmbeddings
from RAG.indexing.vector_store.faiss_index import FaissFlatStore
from RAG.indexing.vector_store.interface import IndexStore


class DenseSearch:
    """
    Dense retriever using embeddings and a vector index.

    This class performs similarity search over a vector index (e.g., FAISS) using dense embeddings.
    """

    def __init__(self, index: IndexStore = FaissFlatStore, embedder: Embeddings = HFEmbeddings):
        """
        Initialize the dense search retriever.

        Args:
            index (IndexStore): Vector index implementing the IndexStore interface (e.g., FaissFlatStore). Used to store and search embeddings.
            embedder (Embeddings): Embedding model implementing the Embeddings interface (e.g., HFEmbeddings). Used to generate query embeddings.
        """
        self.index = index
        self.embedder = embedder

    def dense_search(self, query: str, k: int = 60) -> List[Tuple[int, float]]:
        """
        Perform dense similarity search using the index and embedder.

        The query is first embedded into a dense vector. 
        The vector index is then queried to retrieve the top-k most similar chunks. 
        If embeddings are L2-normalized the inner product corresponds to cosine similarity.

        Args:
            query (str): Query text.
            k (int, optional): Number of top results to return. Defaults to 60.

        Returns:
            List[Tuple[int, float]]: List of (chunk_id, score) tuples, where
                `chunk_id` is the id of the matching chunk in the index, and
                `score` is the similarity score.
        """
        q_vec = self.embedder.embed_query(query)
        D, I = self.index.search(q_vec, k)

        return [(int(idx), float(sim)) for idx, sim in zip(I, D)]