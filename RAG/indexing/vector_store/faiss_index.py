"""IndexStore implementation on FAISS IndexFlatIP + IndexIDMap2."""

from __future__ import annotations
from typing import Tuple, Literal, Optional
from .interface import IndexStore


import numpy as np
import faiss

from .interface import IndexStore


def _ensure_f32_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"vectors must be 2D (N, D), got shape={x.shape}")
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return x


def _ensure_f32_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"qvec must be 1D (D,), got shape={x.shape}")
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return x


def _ensure_i64_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"ids must be 1D (N,), got shape={x.shape}")
    if x.dtype != np.int64:
        x = x.astype(np.int64)
    return x


Device = Literal["cpu", "gpu"]

class FaissFlatStore(IndexStore):
    """
    FAISS-based flat (inner product) index store with optional CPU/GPU support.

    This implementation uses a flat index (`IndexFlatIP`) wrapped with `IndexIDMap2`
    for explicit ID management. It supports storing embeddings, performing similarity
    search, transferring between CPU/GPU, and saving/loading the index.

    Attributes:
        _dim (int): Embedding dimensionality of the index.
        device (str): Execution device ("cpu" or "gpu").
        _gpu_id (int): ID of the GPU device if running on GPU.
        _gpu_res (Optional[faiss.StandardGpuResources]): GPU resources (if allocated).
        index (faiss.Index): Underlying FAISS index instance.
    """

    def __init__(self, dim: int, device: Device = "cpu", gpu_id: int = 0):
        """
        Args:
            dim (int): Embedding dimensionality. Must be > 0.
                This must match the embedding dimension of the vectors inserted into the index!
            device (str, optional): Execution device ("cpu" or "gpu").
                Defaults to "cpu".
            gpu_id (int, optional): GPU device ID (only relevant if device="gpu").
                Defaults to 0.

        Raises:
            ValueError: If `dim <= 0` or if `device` is not "cpu" or "gpu".
        """
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self._dim = int(dim)
        self.device: Device = device
        self._gpu_id = int(gpu_id)
        self._gpu_res: Optional[faiss.StandardGpuResources] = None

        if device == "cpu":
            base = faiss.IndexFlatIP(self._dim)
            self.index = faiss.IndexIDMap2(base)
        elif device == "gpu":
            self._gpu_res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.device = self._gpu_id
            self.index = faiss.GpuIndexFlatIP(self._gpu_res, self._dim, cfg)
        else:
            raise ValueError("device must be 'cpu' or 'gpu'")

    @property
    def dim(self) -> int: 
        """Return the embedding dimensionality of the index."""
        return self._dim

    def size(self) -> int:
        """Return the number of vectors currently stored in store"""
        return int(self.index.ntotal)

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """
        Add embeddings to the index store with explicit IDs.

        Args:
            vectors (np.ndarray): 2D NumPy array of shape (n, dim), containing the embeddings to be added.
            ids (np.ndarray): 1D NumPy array of shape (n,), containing unique integer IDs corresponding to each embedding in `vectors`.
        
        Raises:
            ValueError: If shapes are inconsistent or embedding dimension mismatches.
        """
        vectors = _ensure_f32_2d(vectors)
        ids = _ensure_i64_1d(ids)
        if vectors.shape[0] != ids.shape[0]:
            raise ValueError("vectors and ids must have the same length")
        if vectors.shape[1] != self._dim:
            raise ValueError(f"vectors dim mismatch: got {vectors.shape[1]}, expected {self._dim}")
        self.index.add_with_ids(vectors, ids)

    def search(self, qvec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform similarity search for a single query vector.

        This index uses dot product as the similarity measure.
        If embeddings are L2-normalized before insertion, the search is equivalent to cosine similarity search.

        Args:
            qvec (np.ndarray): 1D float32 array of shape (dim,), the query vector.
            k (int): Number of nearest neighbors to retrieve. Must be > 0.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - scores (np.ndarray): 1D float32 array of shape (k,), similarity scores.
                - ids (np.ndarray): 1D int64 array of shape (k,), IDs of the nearest vectors.

        Raises:
            ValueError: If `k <= 0` or query dimensionality mismatches index dim.
        """
        if k <= 0: raise ValueError("k must be > 0")
        qvec = _ensure_f32_1d(qvec)
        if qvec.shape[0] != self._dim:
            raise ValueError(f"qvec dim mismatch: got {qvec.shape[0]}, expected {self._dim}")
        D, I = self.index.search(qvec[np.newaxis, :], k)
        return D[0], I[0]

    def to_gpu(self, gpu_id: int = 0) -> "FaissFlatStore":
        """
        Transfer the index from CPU to GPU.

        Args:
            gpu_id (int, optional): GPU device ID to use. Defaults to 0.

        Returns:
            FaissFlatStore: The updated instance running on GPU.
        """
        if self.device == "gpu":
            return self
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, self.index)
        self.index = gpu_index
        self._gpu_res = res
        self._gpu_id = int(gpu_id)
        self.device = "gpu"
        return self

    def to_cpu(self) -> "FaissFlatStore":
        """
        Transfer the index from GPU to CPU.

        Returns:
            FaissFlatStore: The updated instance running on CPU.
        """
        if self.device == "cpu":
            return self
        cpu_index = faiss.index_gpu_to_cpu(self.index)
        self.index = cpu_index
        self._gpu_res = None
        self.device = "cpu"
        return self

    def save(self, path: str) -> None:
        """
        Save the index to disk.

        Args:
            path (str): File path where the index should be saved.
        """
        if self.device == "gpu":
            tmp_cpu = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(tmp_cpu, str(path))
        else:
            faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str) -> "FaissFlatStore":
        """
        Load an index from disk (always into CPU memory).

        Args:
            path (str): File path from which to load the index.

        Returns:
            FaissFlatStore: Loaded index instance.
        """
        obj = cls.__new__(cls)
        obj.index = faiss.read_index(str(path))  # CPU
        obj._dim = int(obj.index.d)
        obj.device = "cpu"
        obj._gpu_id = 0
        obj._gpu_res = None
        return obj