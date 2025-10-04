# splitters.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_SEPARATORS: Sequence[str] = ("\n\n", "\n", " ", "")

@dataclass
class SplitterConfig:
    """
    Configuration for document splitting.

    Attributes:
        chunk_size (int): Maximum size (in characters) of each text chunk.
            Defaults to 1000.
        chunk_overlap (int): Number of characters that overlap between
            consecutive chunks. Defaults to 200.
        separators (Sequence[str]): List of separators used to split text,
            applied in priority order. Defaults to ("\n\n", "\n", " ", "").
    """
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    separators: Sequence[str] = DEFAULT_SEPARATORS

class RecursiveSplitter:
    """
    Wrapper around LangChain's RecursiveCharacterTextSplitter.

    This splitter breaks documents into smaller overlapping chunks based on
    a configurable chunk size, overlap, and set of separators.
    """

    def __init__(self, config: Optional[SplitterConfig] = None) -> None:
        """
        Initialize a recursive splitter.

        Args:
            config (Optional[SplitterConfig], optional): Configuration object.
                If None, default values are used.
        """
        self.config = config or SplitterConfig()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=list(self.config.separators),
        )

    def split(self, docs: List[Document]) -> List[Document]:
        """
        Split documents into smaller overlapping chunks.

        Each resulting chunk inherits metadata from the original document and
        adds a `"chunk_id"` field to uniquely identify its position in the
        sequence.

        Args:
            docs (List[Document]): List of input LangChain Document objects.

        Returns:
            List[Document]: List of chunked documents.
        """
        chunks = self._splitter.split_documents(docs)
        for i, ch in enumerate(chunks):
            ch.metadata = {**ch.metadata, "chunk_id": i}
        return chunks
