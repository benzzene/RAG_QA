from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np


class Embeddings(ABC):
    """Interface for creating embeddings."""

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        Return the embedding of a query as a NumPy vector of shape (dim,).

        Args:
            query (str): The query text to embed.

        Returns:
            np.ndarray: The embedding vector of shape (dim,), where
                dim is the embedding dimension of the model.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        ...

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """
        Return the embeddings of multiple texts as a 2D NumPy array of shape (n, dim).

        Args:
            texts (Sequence[str]): List of input texts to embed.

        Returns:
            np.ndarray: 2D array of embeddings with shape (n, dim), where 
                n is the number of texts to embedd,
                dim is the embedding dimension of the model.
        """
        ...

    @abstractmethod
    def is_normalized(self) -> bool:
        """
        Indicate whether the embeddings returned by this model are L2-normalized.

        Returns:
            bool: True if embeddings are normalized, False otherwise.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        ...

    def dim(self) -> int:
        """
        Return the dimensionality of the embeddings.

        Returns:
            int: The embedding dimension (e.g., 1024 for BGE-M3).
        """
        ...
