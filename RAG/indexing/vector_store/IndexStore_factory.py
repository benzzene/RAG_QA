from typing import Mapping, Any, Optional
from .interface import IndexStore
from .faiss_index import FaissFlatStore  

def create_indexstore(cfg: Mapping[str, Any], *, embedder: Optional[Any] = None) -> IndexStore:
    """
    Create an index store instance based on the given configuration.

    Currently only the FAISS flat index store (`FaissFlatStore`) is supported.

    Args:
        cfg (Mapping[str, Any]): Configuration dictionary with the following keys:
            - provider (str, optional): Index provider. Must be "faiss_flat".
              Defaults to "faiss_flat".
            - device (str, optional): Execution device, either "cpu" or "gpu".
              Defaults to "gpu".
            - gpu_id (int, optional): GPU device ID if using GPU execution.
              Defaults to 0.
            - dim (int, optional): Embedding dimensionality. If not provided,
              `embedder` must be passed with a `.dim()` method.

        embedder (Optional[Any], optional): Embedding model providing a `dim()`
            method. Used to determine dimensionality if `cfg["dim"]` is not set.

    Returns:
        IndexStore: An initialized `FaissFlatStore` instance.

    Raises:
        ValueError: If the provider is unknown, or if the dimension cannot be
            determined from either `cfg["dim"]` or `embedder.dim()`.
    """
    provider = str(cfg.get("provider", "faiss_flat")).lower()

    if provider != "faiss_flat":
        raise ValueError(f"Unknown provider: {provider}")

    device = str(cfg.get("device", "gpu")).lower()   
    gpu_id = int(cfg.get("gpu_id", 0))
    dim = cfg.get("dim")
    if dim is None:
        if embedder is None or not hasattr(embedder, "dim"):
            raise ValueError("Brak 'dim' w cfg i brak embeddera. Podaj cfg['dim'] albo przekaż embedder z metodą dim().")
        dim = int(embedder.dim())

    return FaissFlatStore(dim=int(dim), device=device, gpu_id=gpu_id)
