from torch import nn

from rldx.configs.execution import DEFAULT_HIDDEN_ALIGNMENT
from rldx.model.modules.backbone.modeling_qwen3_vl import Qwen3VLVisionMLP
from rldx.model.modules.hidden_padding import pad_hidden


class PaddedVisionMLP(Qwen3VLVisionMLP):
    def __init__(self, mlp: Qwen3VLVisionMLP, alignment: int) -> None:
        nn.Module.__init__(self)
        self.hidden_size = mlp.hidden_size
        self.linear_fc1 = mlp.linear_fc1
        self.linear_fc2 = mlp.linear_fc2
        self.act_fn = mlp.act_fn
        pad_hidden(self, "linear_fc1", "linear_fc2", alignment)

    @property
    def logical_hidden_features(self) -> int:
        return self.hidden_layout.logical

    @property
    def intermediate_size(self) -> int:
        return self.linear_fc2.in_features


def pad_vision_mlps(visual: nn.Module, alignment: int = DEFAULT_HIDDEN_ALIGNMENT) -> None:
    for block in visual.blocks:
        block.mlp = PaddedVisionMLP(block.mlp, alignment)
