"""The static heap boundary: what the collector is allowed to traverse.

CPython's collector walks every tracked container on a full collection, so a
training process pays for its static heap -- modules, model graph, closures --
on every gen-2 pass.  ``gc.freeze()`` moves that heap into the permanent
generation, which the collector never walks.

The freeze is not a one-off.  It holds only for the objects alive when it runs,
so anything that repopulates the heap afterwards lands outside the frozen set
and is walked again on every full collection from then on.
"""

from __future__ import annotations

import gc


def freeze_static_heap() -> None:
    """Take the live heap out of the collector's reach.

    Unfreezing first keeps the permanent generation at the live set: freezing
    on top of a previous freeze would grow it by one window's survivors each
    time, and nothing in it is ever reclaimed.
    """
    gc.unfreeze()
    gc.collect()
    gc.freeze()


def unfreeze_worker_heap(_worker_id: int) -> None:
    """Hand the collector back the heap a loader worker inherited at fork.

    A worker forks after the parent has frozen, so it starts with a permanent
    generation it never chose.  Left frozen, the worker's own full collections
    reclaim nothing and its data supply stalls.
    """
    gc.unfreeze()
