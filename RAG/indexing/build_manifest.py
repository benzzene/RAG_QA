from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from RAG.config import IndexConfig


MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    created_at: str
    docs_dir: str
    files: list[dict[str, Any]]
    embedding_provider: str
    embedding_model: str
    embedding_normalize: bool
    similarity_metric: str
    chunk_size: int
    chunk_overlap: int


def build_manifest(cfg: IndexConfig) -> Manifest:
    """Build a manifest describing the current index inputs.

    Args:
        cfg: Typed configuration for index building.

    Returns:
        Manifest describing the current source documents and indexing settings.
    """
    docs_dir = Path(cfg.docs_dir)

    return Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        docs_dir=str(docs_dir),
        files=_collect_files(docs_dir),
        embedding_provider=cfg.embeddings.provider,
        embedding_model=cfg.embeddings.model,
        embedding_normalize=cfg.embeddings.normalize,
        similarity_metric=cfg.embeddings.similarity_metric,
        chunk_size=cfg.splitter.chunk_size,
        chunk_overlap=cfg.splitter.chunk_overlap,
    )


def save_manifest(path: str | Path, manifest: Manifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_manifest(path: str | Path) -> Manifest | None:
    path = Path(path)

    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    return Manifest(**data)


def manifest_matches(saved: Manifest | None, current: Manifest) -> bool:
    if saved is None:
        return False

    return _comparable(saved) == _comparable(current)


def manifest_diff(saved: Manifest | None, current: Manifest) -> list[str]:
    if saved is None:
        return ["manifest file is missing"]

    reasons: list[str] = []

    if saved.schema_version != current.schema_version:
        reasons.append("manifest schema version changed")

    if saved.docs_dir != current.docs_dir:
        reasons.append("docs directory changed")

    if saved.embedding_provider != current.embedding_provider:
        reasons.append("embedding provider changed")

    if saved.embedding_model != current.embedding_model:
        reasons.append("embedding model changed")

    if saved.embedding_normalize != current.embedding_normalize:
        reasons.append("embedding normalize setting changed")

    if saved.similarity_metric != current.similarity_metric:
        reasons.append("similarity metric changed")

    if saved.chunk_size != current.chunk_size:
        reasons.append("chunk size changed")

    if saved.chunk_overlap != current.chunk_overlap:
        reasons.append("chunk overlap changed")

    saved_files = {f["path"]: f for f in saved.files}
    current_files = {f["path"]: f for f in current.files}

    added = sorted(set(current_files) - set(saved_files))
    removed = sorted(set(saved_files) - set(current_files))
    common = sorted(set(saved_files) & set(current_files))

    if added:
        reasons.append(f"new source files: {', '.join(added)}")

    if removed:
        reasons.append(f"removed source files: {', '.join(removed)}")

    changed = [
        path
        for path in common
        if saved_files[path] != current_files[path]
    ]

    if changed:
        reasons.append(f"changed source files: {', '.join(changed)}")

    return reasons


def _comparable(manifest: Manifest) -> dict[str, Any]:
    data = asdict(manifest)
    data.pop("created_at", None)
    return data


def _collect_files(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        raise FileNotFoundError(f"Docs directory does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Docs path is not a directory: {root}")

    files: list[dict[str, Any]] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        stat = path.stat()

        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _sha256_file(path),
            }
        )

    return files


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()