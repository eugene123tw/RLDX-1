from collections import OrderedDict
from typing import Callable, Hashable, TypeVar


Value = TypeVar("Value")


def lru_get(cache: OrderedDict, key: Hashable, size: int, compute: Callable[[], Value]) -> Value:
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    value = cache[key] = compute()
    if len(cache) > size:
        cache.popitem(last=False)
    return value
