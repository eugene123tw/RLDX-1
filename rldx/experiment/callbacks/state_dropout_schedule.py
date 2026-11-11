"""Anneal model.state_dropout_prob during training.

Forward pass at ``rldx/model/core/rldx.py:348-352`` reads
``self.state_dropout_prob`` every step, so all this callback has to do
is overwrite that attribute on the live model before each optimizer
step. ``RLDX.__init__`` sets the initial value from
``config.state_dropout_prob`` (the unfinished start of the schedule);
we then drive it down to ``state_dropout_prob_final`` over the run.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.modeling_utils import unwrap_model


logger = logging.getLogger(__name__)


def _cosine(start: float, end: float, progress: float) -> float:
    """Half-cosine from ``start`` (progress=0) to ``end`` (progress=1)."""
    progress = max(0.0, min(1.0, progress))
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def _linear(start: float, end: float, progress: float) -> float:
    progress = max(0.0, min(1.0, progress))
    return start + (end - start) * progress


_SCHEDULES = {"cosine": _cosine, "linear": _linear}


class StateDropoutScheduleCallback(TrainerCallback):
    """Drives ``model.state_dropout_prob`` along ``schedule`` every step.

    Args:
        start: Initial dropout probability (progress=0). Usually the
            ``--state-dropout-prob`` CLI value.
        end:   Target dropout probability (progress=1). Usually the
            ``--state-dropout-prob-final`` CLI value.
        max_steps: Total optimizer steps for the run; progress is
            ``state.global_step / max_steps``.
        schedule: ``"cosine"`` (default) or ``"linear"``.
        log_every: Emit one info-level log line every N optimizer steps
            (rank-0 only) so the curve is visible in stdout / wandb.
    """

    def __init__(
        self,
        start: float,
        end: float,
        max_steps: int,
        schedule: str = "cosine",
        log_every: int = 200,
    ) -> None:
        if schedule not in _SCHEDULES:
            raise ValueError(
                f"StateDropoutScheduleCallback: unknown schedule '{schedule}'. "
                f"Choose from {sorted(_SCHEDULES)}."
            )
        if max_steps <= 0:
            raise ValueError(
                f"StateDropoutScheduleCallback: max_steps must be > 0 (got {max_steps})."
            )
        self.start = float(start)
        self.end = float(end)
        self.max_steps = int(max_steps)
        self.schedule = schedule
        self._fn = _SCHEDULES[schedule]
        self.log_every = max(1, int(log_every))

    def _current_value(self, global_step: int) -> float:
        progress = global_step / self.max_steps
        return self._fn(self.start, self.end, progress)

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        model = kwargs.get("model")
        if model is None:
            return control
        raw = unwrap_model(model)
        new_val = self._current_value(state.global_step)
        try:
            raw.state_dropout_prob = float(new_val)
        except Exception as exc:
            logger.warning("StateDropoutScheduleCallback: set failed: %s", exc)
            return control
        if state.global_step % self.log_every == 0:
            logger.info(
                "StateDropoutScheduleCallback step=%d state_dropout_prob=%.4f "
                "(schedule=%s start=%.3f end=%.3f)",
                state.global_step,
                new_val,
                self.schedule,
                self.start,
                self.end,
            )
        return control
