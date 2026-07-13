from pathlib import Path
from typing import Collection


def make_unique_document_name(name: str, existing_names: Collection[str]) -> str:
    """Return a stable display name even when uploads share the same filename."""
    if name not in existing_names:
        return name

    path = Path(name)
    counter = 2
    while True:
        candidate = f"{path.stem} ({counter}){path.suffix}"
        if candidate not in existing_names:
            return candidate
        counter += 1


def upload_limit_violation(
    file_sizes: Collection[int],
    max_files: int,
    max_total_bytes: int,
) -> str | None:
    """Return a stable violation code for public-demo upload guardrails."""
    if len(file_sizes) > max_files:
        return "too_many_files"
    if sum(file_sizes) > max_total_bytes:
        return "total_size_exceeded"
    return None
