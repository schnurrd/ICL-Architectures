from __future__ import annotations

from typing import Any, Iterable

import torch


def iter_named_tensors(
    obj: Any,
    *,
    prefix: str = "state",
    visited: set[int] | None = None,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield (name, tensor) pairs from nested Python/object structures.

    Traversal is cycle-safe via `visited` object-id tracking.
    """
    if visited is None:
        visited = set()

    oid = id(obj)
    if oid in visited:
        return
    visited.add(oid)

    if torch.is_tensor(obj):
        yield prefix, obj
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_named_tensors(value, prefix=child, visited=visited)
        return

    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            child = f"{prefix}[{idx}]"
            yield from iter_named_tensors(value, prefix=child, visited=visited)
        return

    if hasattr(obj, "__dict__"):
        for key, value in vars(obj).items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_named_tensors(value, prefix=child, visited=visited)
