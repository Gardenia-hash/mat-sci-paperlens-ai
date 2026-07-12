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
