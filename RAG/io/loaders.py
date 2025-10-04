from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Set
from tqdm.auto import tqdm
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)


try:
    from langchain_community.document_loaders import UnstructuredFileLoader
    HAS_UNSTRUCTURED = True
except Exception:
    HAS_UNSTRUCTURED = False


DEFAULT_EXTENSIONS: Set[str] = {".pdf", ".txt", ".md"}
if HAS_UNSTRUCTURED:
    DEFAULT_EXTENSIONS |= {".docx", ".html", ".htm"}


@dataclass
class LoaderConfig:
    """
    Configuration for directory-based document loading.

    Attributes:
        root_dir (str | os.PathLike): Root directory to recursively scan for files.
            Defaults to "docs".
        extensions (Set[str]): File extensions to include (e.g., {".pdf", ".txt"}).
            Defaults to supported extensions depending on installed libraries.
        show_progress (bool): Whether to show a progress bar during loading.
            Defaults to True.
        add_source_path (bool): Whether to add the file path to each document's
            metadata under the key `"source_path"`. Defaults to True.
    """
    root_dir: str | os.PathLike = "docs"
    extensions: Set[str] = field(default_factory=lambda: set(DEFAULT_EXTENSIONS))
    show_progress: bool = True
    add_source_path: bool = True 

class FileLoaderFactory:
    """Factory for selecting the appropriate LangChain document loader for a single file."""
    
    def __init__(self, has_unstructured: bool = HAS_UNSTRUCTURED) -> None:
        """
        Initialize the factory.

        Args:
            has_unstructured (bool, optional): Whether unstructured loaders
                (docx, html) are available. Defaults to global HAS_UNSTRUCTURED.
        """
        self.has_unstructured = has_unstructured

    def make(self, path: Path):
        """
        Return a suitable LangChain loader for the given file path.

        Args:
            path (Path): Path to the file.

        Returns:
            BaseLoader | None: A LangChain loader instance if the file extension
            is supported, otherwise None.
        """
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return PyPDFLoader(str(path))
        if suffix == ".txt":
            return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
        if suffix == ".md":
            return UnstructuredMarkdownLoader(str(path))
        if self.has_unstructured and suffix in {".docx", ".html", ".htm"}:
            return UnstructuredFileLoader(str(path))
        return None


class DirectoryDocumentLoader:
    """
    Load documents from a directory recursively without chunking.

    This loader iterates over files with supported extensions, applies
    the appropriate loader for each file, and optionally adds the file
    path to document metadata.
    """
    def __init__(self, config: LoaderConfig, factory: Optional[FileLoaderFactory] = None) -> None:
        """
        Args:
            config (LoaderConfig): Configuration object controlling loading behavior.
            factory (Optional[FileLoaderFactory], optional): Factory for creating
                file loaders. If None, a default FileLoaderFactory is used.
        """
        self.config = config
        self.factory = factory or FileLoaderFactory()

    def load(self) -> List[Document]:
        """
        Load all documents from the configured root directory.

        Returns:
            List[Document]: List of LangChain Document objects loaded from files.
        """
        files = self._collect_files()
        docs: List[Document] = []
        iterator = tqdm(files, desc="Loading files", unit="file") if self.config.show_progress else files

        for path in iterator:
            loader = self.factory.make(path)
            if loader is None:
                continue
            try:
                loaded = loader.load()
                if self.config.add_source_path:
                    for d in loaded:
                        d.metadata = {**d.metadata, "source_path": str(path)}
                docs.extend(loaded)
            except Exception as e:
                print(f"[WARN] Skipping {path.name}: {e}")
        return docs


    def _collect_files(self) -> List[Path]:
        """
        Collect all files from the root directory matching allowed extensions.

        Returns:
            List[Path]: List of file paths.

        Raises:
            FileNotFoundError: If the root directory does not exist or is not a directory.
        """
        exts = set(e.lower() for e in (self.config.extensions or DEFAULT_EXTENSIONS))
        root = Path(self.config.root_dir)
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Directory doesn't exist or isn't a directory: {root}")
        return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
