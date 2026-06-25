from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SplitterConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "SplitterConfigModel":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class EmbeddingsConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["hf"] = "hf"
    model: str = "BAAI/bge-m3"
    device: Literal["cpu", "cuda"] = "cpu"
    normalize: bool = True
    batch_size: int = Field(default=32, gt=0)
    similarity_metric: str = "cosine"


class IndexStoreConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["faiss_flat"] = "faiss_flat"
    device: Literal["cpu", "gpu"] = "cpu"
    gpu_id: int = Field(default=0, ge=0)


class IndexConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    docs_dir: str = "docs"
    index_path: str = "data/index.faiss"
    chunks_path: str = "data/chunks.jsonl"
    show_progress: bool = True
    splitter: SplitterConfigModel = Field(default_factory=SplitterConfigModel)
    embeddings: EmbeddingsConfigModel = Field(default_factory=EmbeddingsConfigModel)
    store: IndexStoreConfigModel = Field(default_factory=IndexStoreConfigModel)


class RetrieverConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k_bm25: int = Field(default=60, gt=0)
    k_dense: int = Field(default=60, gt=0)
    topn: int = Field(default=40, gt=0)


class RerankerConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: Literal["cpu", "cuda"] = "cpu"
    max_length: int = Field(default=512, gt=0)
    k_final: int = Field(default=5, gt=0)
    batch_size: int = Field(default=32, gt=0)


class LLMConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["qwen"] = "qwen"
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    device: Literal["cpu", "cuda"] = "cpu"
    use_4bit: bool = True
    system_prompt: str = (
        "Jesteś asystentem QA. Odpowiadasz wyłącznie na podstawie przekazanego kontekstu. "
        "Twoim zadaniem jest wydobyć informację z kontekstu, nawet jeżeli są rozproszone. "
        "Cytuj źródła, gdy to możliwe."
    )
    max_context_chars: int = Field(default=12000, gt=0)
    max_new_tokens: int = Field(default=512, gt=0)
    temperature: float = Field(default=0.2, ge=0.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    do_sample: bool = True
    repetition_penalty: float = Field(default=1.1, gt=0.0)
    num_beams: int = Field(default=1, gt=0)
    num_return_sequences: int = Field(default=1, gt=0)


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_path: str = "data/index.faiss"
    chunks_path: str = "data/chunks.jsonl"
    build_if_missing: bool = True
    embeddings: EmbeddingsConfigModel = Field(default_factory=EmbeddingsConfigModel)
    retriever: RetrieverConfigModel = Field(default_factory=RetrieverConfigModel)
    reranker: RerankerConfigModel = Field(default_factory=RerankerConfigModel)
    llm: LLMConfigModel = Field(default_factory=LLMConfigModel)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: IndexConfig = Field(default_factory=IndexConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def validate_paths(self) -> "AppConfig":
        if self.index.index_path != self.runtime.index_path:
            raise ValueError("index.index_path must match runtime.index_path")
        if self.index.chunks_path != self.runtime.chunks_path:
            raise ValueError("index.chunks_path must match runtime.chunks_path")
        return self


def load_config(path: str | Path = "config/default_config.yaml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    return AppConfig.model_validate(payload)
