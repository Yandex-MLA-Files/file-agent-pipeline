from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class DocumentAssetStore(Protocol):
    """Provides original uploaded file bytes to runtime-only document tools."""

    def get_bytes(self, source_file: str) -> bytes:
        """Return the original bytes for an indexed source file."""
        ...


@dataclass
class InMemoryDocumentAssetStore:
    """Process-local asset storage for Streamlit and one-shot RAG calls."""

    _assets: dict[str, bytes] = field(default_factory=dict)

    @classmethod
    def from_files(cls, file_paths: list[str | Path]) -> "InMemoryDocumentAssetStore":
        store = cls()
        for file_path in file_paths:
            path = Path(file_path)
            store.put(path.name, path.read_bytes())
        return store

    def put(self, source_file: str, contents: bytes) -> None:
        file_name = Path(source_file).name.strip()
        if not file_name:
            raise ValueError("source_file must not be empty")
        if not contents:
            raise ValueError(f"document asset is empty: {file_name}")

        duplicate = next(
            (name for name in self._assets if name.casefold() == file_name.casefold()),
            None,
        )
        if duplicate is not None:
            raise ValueError(f"multiple uploaded documents share this file name: {file_name}")
        self._assets[file_name] = bytes(contents)

    def get_bytes(self, source_file: str) -> bytes:
        requested_name = Path(source_file).name.strip().casefold()
        if not requested_name:
            raise ValueError("source_file must not be empty")

        matches = [
            contents
            for file_name, contents in self._assets.items()
            if file_name.casefold() == requested_name
        ]
        if not matches:
            raise ValueError(f"original document asset is not available: {source_file}")
        if len(matches) > 1:  # Defensive for custom/populated mappings.
            raise ValueError(f"multiple document assets share this file name: {source_file}")
        return matches[0]

    def clear(self) -> None:
        self._assets.clear()
