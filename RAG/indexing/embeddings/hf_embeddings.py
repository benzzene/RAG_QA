from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import torch 
from typing import Sequence
from .embeddings import Embeddings

class HFEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "sentence-transformers/BAAI/bge-m3",
        normalize: bool = True,
        batch_size: int = 32,
        dtype=np.float32,
        similarity_metric: str = "cosine",
        device: str | None = None,
    ):
        """
        Initialize the embedding model.

        Args:
            model_name (str, optional): The Hugging Face model to load.
            normalize (bool, optional): Whether to L2-normalize embeddings.
            batch_size (int, optional): Batch size for document embeddings.
            dtype (np.dtype, optional): Data type for returned arrays.
            similarity_metric (str, optional): Similarity metric associated with the embeddings.
            device (str | None, optional): Device to run the model on (`"cpu"`, `"cuda"`).
                If `None`, automatically selects `"cuda"` if available, otherwise `"cpu"`.

        Attributes:
            model (SentenceTransformer): The underlying transformer model.
            dimension (int): Embedding dimension of the model.
        """
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self.dtype = dtype
        self.similarity_metric = similarity_metric
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_query(self, query: str) -> np.ndarray:
        """
        Return the embedding of a query as a NumPy vector of shape (dim,).

        Args:
            query (str): Input query text.

        Returns:
            np.ndarray: Embedding vector of shape (dim,), cast to `self.dtype`, where
                dim is the embedding dimension of the model.
        """
        vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        return vec[0].astype(self.dtype)

    def embed_documents(self, texts: Sequence[str], show_progress: bool = True) -> np.ndarray:
        """
        Return the embeddings of multiple texts as a 2D NumPy array of shape (n, dim).

        Each document should first be split into a sequence of texts (chunks),
        and each chunk is then embedded into a dense vector.
        
        Args:
            texts (Sequence[str]): List of input texts to embed.
            show_progress (bool, optional): Whether to show a progress bar during batch embedding.

        Returns:
            np.ndarray: 2D array of embeddings with shape (n, dim), where 
                n is the number of texts (chunks) to embedd,
                dim is the embedding dimension of the model.
        """
        embeddings = []
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding documents")

        for i in iterator:
            batch = texts[i:i+self.batch_size]
            vecs = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=True
            )
            embeddings.append(vecs.astype(self.dtype))

        return np.vstack(embeddings)
    
    def is_normalized(self) -> bool:
        """
        Indicate whether embeddings are L2-normalized.

        Returns:
            bool: True if embeddings are normalized, False otherwise.
        """
        return self.normalize

    def dim(self) -> int:
        """
        Return the dimensionality of embeddings.

        Returns:
            int: Embedding dimension (e.g., 1024 for BGE-M3).
        """
        return int(self.dimension)
    
    def similarity(self) -> str:
        """
        Return the similarity metric associated with these embeddings.

        Returns:
            str: The similarity metric.
        """
        return str(self.similarity_metric)