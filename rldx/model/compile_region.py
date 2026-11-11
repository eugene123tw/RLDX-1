from dataclasses import dataclass
from typing import Callable, Literal

import torch


@dataclass(frozen=True)
class CompileRegion:
    dynamic: bool | None
    autotune: bool
    backend: Literal["inductor", "aot_eager"] = "inductor"

    @property
    def mode(self) -> str:
        return "max-autotune-no-cudagraphs" if self.autotune else "default"

    def compile(self, fn: Callable) -> Callable:
        target = {"mode": self.mode} if self.backend == "inductor" else {"backend": self.backend}
        return torch.compile(fn, dynamic=self.dynamic, fullgraph=True, **target)
