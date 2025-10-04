from typing import Mapping, Any
import numpy as np
from .embeddings import Embeddings
from .hf_embeddings import HFEmbeddings


def create_embeddings(cfg: Mapping[str, Any]) -> Embeddings:
    """
    Create an embedding model instance based on the given configuration.

    This factory function initializes an embedding provider (currently only
    Hugging Face `HFEmbeddings` is supported) with the parameters specified
    in the configuration mapping.

    Args:
        cfg (Mapping[str, Any]): Configuration dictionary with the following keys:
            - provider (str, optional): Embedding provider. 
            - model (str, optional): Model name to load. 
            - normalize (bool, optional): Whether to L2-normalize embeddings. 
            - batch_size (int, optional): Batch size for document embeddings. 
            - dtype (np.dtype, optional): Data type for returned embeddings. 
            - similarity_metric (str, optional): Similarity metric (e.g., "cosine").
            - device (str, optional): Device to run the model on ("cpu" or "cuda"). 

    Returns:
        Embeddings: An embedding model instance (currently `HFEmbeddings`).

    Raises:
        ValueError: If an unknown provider is specified in the configuration.
    """
    provider = cfg.get("provider", "hf")

    if provider == "hf":
        return HFEmbeddings(
            model_name=cfg.get("model", "BAAI/bge-m3"),
            normalize=cfg.get("normalize", True),
            batch_size=cfg.get("batch_size", 32),
            dtype=cfg.get("dtype", np.float32),
            similarity_metric=cfg.get("similarity_metric", "cosine"),
            device=cfg.get("device", "gpu")
        )

    raise ValueError(f"Unknown provider: {provider}")
