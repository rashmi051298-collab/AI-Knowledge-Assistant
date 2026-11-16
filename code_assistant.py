"""Local codebase indexing and semantic search."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".jsx",
    ".json",
    ".kt",
    ".md",
    ".php",
    ".phtml",
    ".py",
    ".rb",
    ".rs",
    ".scss",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
}
EXCLUDED_DIRECTORIES = {
    ".code-assistant-index",
    ".git",
    ".hg",
    ".idea",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "storage",
    "vendor",
    "venv",
}
EXCLUDED_FILES = {
    ".env",
    "composer.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
}
MAX_FILE_BYTES = 1_000_000
INDEX_VERSION = 1


@dataclass(frozen=True)
class CodeChunk:
    text: str
    source: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class SearchResult:
    chunk: CodeChunk
    score: float


def should_index_file(path: Path) -> bool:
    """Return whether a relative source path is eligible for indexing."""
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return (
        path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not parts.intersection(EXCLUDED_DIRECTORIES)
        and name not in EXCLUDED_FILES
        and not name.startswith(".env.")
    )


class CodebaseIndex:
    """Build and query a cached FAISS index for a source directory."""

    def __init__(
        self,
        root: Path,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.root = root.expanduser().resolve()
        self.embedding_model_name = embedding_model
        self.cache_dir = self.root / ".code-assistant-index"
        self._index: Any = None
        self._chunks: list[CodeChunk] = []
        self._model: Any = None

    def _read_sources(self) -> tuple[list[tuple[str, str]], str]:
        if not self.root.is_dir():
            raise ValueError(f"Codebase folder does not exist: {self.root}")

        sources: list[tuple[str, str]] = []
        fingerprint = hashlib.sha256()
        for current_root, directories, filenames in os.walk(self.root):
            directories[:] = sorted(
                name
                for name in directories
                if name.lower() not in EXCLUDED_DIRECTORIES
            )
            for filename in sorted(filenames):
                path = Path(current_root) / filename
                relative_path = path.relative_to(self.root)
                if path.is_symlink() or not should_index_file(relative_path):
                    continue
                try:
                    if path.stat().st_size > MAX_FILE_BYTES:
                        continue
                    raw_content = path.read_bytes()
                except OSError as error:
                    raise OSError(
                        f"Could not read source file {relative_path}: {error}"
                    ) from error
                if b"\x00" in raw_content:
                    continue
                content = raw_content.decode("utf-8", errors="replace")
                if not content.strip():
                    continue

                source = relative_path.as_posix()
                fingerprint.update(source.encode("utf-8"))
                fingerprint.update(b"\0")
                fingerprint.update(raw_content)
                fingerprint.update(b"\0")
                sources.append((source, content))

        fingerprint.update(self.embedding_model_name.encode("utf-8"))
        return sources, fingerprint.hexdigest()

    def build(self) -> int:
        """Load a matching cache or embed and persist the current codebase."""
        import faiss
        import numpy as np
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from sentence_transformers import SentenceTransformer

        if not self.root.is_dir():
            raise ValueError(f"Codebase folder does not exist: {self.root}")

        sources, fingerprint = self._read_sources()
        self._model = SentenceTransformer(self.embedding_model_name)
        index_path = self.cache_dir / "index.faiss"
        metadata_path = self.cache_dir / "chunks.json"

        if index_path.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("version") == INDEX_VERSION
                and metadata.get("fingerprint") == fingerprint
            ):
                self._index = faiss.read_index(str(index_path))
                self._chunks = [CodeChunk(**item) for item in metadata["chunks"]]
                return len(self._chunks)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=180,
            add_start_index=True,
        )
        chunks: list[CodeChunk] = []
        for source, content in sources:
            documents = splitter.create_documents([content], metadatas=[{"source": source}])
            for document in documents:
                start = document.metadata.get("start_index", 0)
                start_line = content.count("\n", 0, start) + 1
                end_offset = start + len(document.page_content)
                end_line = content.count("\n", 0, end_offset) + 1
                if document.page_content.endswith("\n"):
                    end_line -= 1
                chunks.append(
                    CodeChunk(
                        text=document.page_content,
                        source=source,
                        start_line=start_line,
                        end_line=max(start_line, end_line),
                    )
                )

        if not chunks:
            raise ValueError(
                "No indexable source code found. Check the folder and supported file types."
            )

        vectors = self._model.encode(
            [chunk.text for chunk in chunks],
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_index = self.cache_dir / "index.faiss.tmp"
        temporary_metadata = self.cache_dir / "chunks.json.tmp"
        faiss.write_index(index, str(temporary_index))
        temporary_index.replace(index_path)
        temporary_metadata.write_text(
            json.dumps(
                {
                    "version": INDEX_VERSION,
                    "fingerprint": fingerprint,
                    "chunks": [asdict(chunk) for chunk in chunks],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary_metadata.replace(metadata_path)

        self._index = index
        self._chunks = chunks
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return the most similar code chunks for a natural-language query."""
        import numpy as np
        from sentence_transformers import SentenceTransformer

        if not query.strip():
            raise ValueError("Enter a question or search query.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if self._index is None or not self._chunks:
            raise ValueError("Index the codebase before searching.")
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)

        query_vector = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores, positions = self._index.search(
            np.ascontiguousarray(query_vector, dtype=np.float32),
            min(top_k, len(self._chunks)),
        )
        return [
            SearchResult(chunk=self._chunks[int(position)], score=float(score))
            for score, position in zip(scores[0], positions[0])
            if 0 <= position < len(self._chunks)
        ]
