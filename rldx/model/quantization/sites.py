from dataclasses import dataclass
from typing import Callable

from torch import nn

from rldx.configs.execution import QuantGroup


@dataclass(frozen=True)
class LinearSite:
    group: QuantGroup
    owner: nn.Module
    name: str
    elementwise_input: bool = False
    input_bound: Callable[[], float] | None = None
