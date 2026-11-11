from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from rldx.data.state_action.norm_stats import NormStats
from rldx.data.types import ActionConfig
from rldx.data.utils import (
    normalize_values_meanstd,
    normalize_values_minmax,
    unnormalize_values_meanstd,
    unnormalize_values_minmax,
)


@dataclass(frozen=True)
class ActionKeyCodec:
    key: str
    embodiment_tag: str
    params: NormStats
    mean_std: bool
    clip: bool
    relative: ActionConfig | None
    to_relative: Callable[..., np.ndarray]
    to_absolute: Callable[..., np.ndarray]

    def encode(self, values, state=None, position=None):
        if self.relative is not None:
            reference = self._reference(state, "relative action processing")
            values = self.to_relative(
                action=values,
                reference_state=reference[-1],
                action_type=self.relative.type,
                action_format=self.relative.format,
            )
        normalize = normalize_values_meanstd if self.mean_std else normalize_values_minmax
        normalized = normalize(values, self._params(position))
        return np.clip(normalized, -1.0, 1.0) if self.clip else normalized

    def decode(self, values, state=None, position=None):
        unnormalize = unnormalize_values_meanstd if self.mean_std else unnormalize_values_minmax
        values = unnormalize(values, self._params(position))
        if self.relative is None:
            return values
        reference = self._reference(state, "relative->absolute conversion")
        if reference.ndim == 2:
            reference = reference[None]
        samples = values if values.ndim == 3 else values[None]
        absolute = np.stack(
            [
                self.to_absolute(
                    action=sample,
                    reference_state=frame[-1],
                    action_type=self.relative.type,
                    action_format=self.relative.format,
                )
                for frame, sample in zip(reference, samples)
            ]
        )
        return absolute if values.ndim == 3 else absolute[0]

    def reframe(self, values, prev_state, new_state, prev_position, new_position):
        if self.relative is None and not self._per_step:
            return values
        return self.encode(self.decode(values, prev_state, prev_position), new_state, new_position)

    @property
    def _per_step(self):
        return self.params.per_step

    def _params(self, position):
        return self.params if position is None else self.params.at(position)

    def _reference(self, state, purpose):
        if state is None:
            raise ValueError(
                f"State dict required for {purpose} of key '{self.key}' "
                f"in embodiment '{self.embodiment_tag}'"
            )
        state_key = self.relative.state_key or self.key
        if state_key not in state:
            raise KeyError(
                f"Reference state key '{state_key}' not found in state dict "
                f"for embodiment '{self.embodiment_tag}'"
            )
        return state[state_key]
