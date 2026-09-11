"""Protocol helpers shared by full-sort recommendation evaluators."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Set


def evaluation_history_items(
    user_id: int,
    prefix: Iterable[int],
    target: int,
    seen_items: Optional[Dict[int, Sequence[int]]] = None,
    num_items: Optional[int] = None,
) -> Set[int]:
    """Return catalog items that must be masked for one evaluation record.

    The actual evaluation prefix is authoritative. ``seen_items`` may add
    older history for compatibility, but it must never replace the prefix.
    The ground-truth target is excluded so repeated-item evaluation remains
    well-defined.
    """
    merged = {int(item) for item in prefix}
    if seen_items is not None:
        merged.update(int(item) for item in seen_items.get(int(user_id), ()))

    upper = int(num_items) if num_items is not None else None
    return {
        item
        for item in merged
        if item > 0
        and item != int(target)
        and (upper is None or item <= upper)
    }

