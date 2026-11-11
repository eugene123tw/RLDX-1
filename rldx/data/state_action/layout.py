import numpy as np


Layout = dict[str, int]


def split(flat: np.ndarray, layout: Layout) -> dict[str, np.ndarray]:
    parts, start = {}, 0
    for key, width in layout.items():
        parts[key] = flat[..., start : start + width]
        start += width
    return parts


def join(parts: dict[str, np.ndarray], layout: Layout) -> np.ndarray:
    return np.concatenate([parts[key] for key in layout], axis=-1)
