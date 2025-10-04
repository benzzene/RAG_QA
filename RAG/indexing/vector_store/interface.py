from __future__ import annotations
from typing import Protocol, Tuple
import numpy as np


class IndexStore(Protocol):
    """
    Interface for creating index store (vector store)

    An index store is responsible for holding vector embeddings and 
    enabling efficient similarity search or retrieval.
    """

    @property
    def dim(self) -> int:
        """
        Return the embedding dimensionality of the index.

        This must match the embedding dimension of the vectors inserted into the index.
        """
        ...

    def size(self) -> int:
        """
        Return the number of vectors currently stored in the index.

        Returns:
            int: Total number of vectors in the index.
        """
        ...

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """
        Add embeddings to the index store with explicit IDs.

        Args:
            vectors (np.ndarray): 2D NumPy array of shape (n, dim), containing the embeddings to be added.
            ids (np.ndarray): 1D NumPy array of shape (n,), containing unique integer IDs corresponding to each embedding in `vectors`.
        """
        ...

    def search(self, qvec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform similarity search for a single query vector.

        Args:
            qvec (np.ndarray): Query vector of shape (dim,).
            k (int): Number of nearest neighbors to retrieve.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - scores (np.ndarray): 1D array of shape (k,), containing similarity
                  scores or distances.
                - ids (np.ndarray): 1D array of shape (k,), containing IDs of the
                  nearest vectors.
        """

    def save(self, path: str) -> None:
        """
        Save index to the disc.

        Args:
            path (str): Path where the index should be saved.
        """
        ...

    @classmethod
    def load(cls, path: str) -> "IndexStore":
        """
        Load an index from disk.

        Args:
            path (str): File path from which the index should be loaded.

        Returns:
            IndexStore: Loaded index instance.
        """
        ...


