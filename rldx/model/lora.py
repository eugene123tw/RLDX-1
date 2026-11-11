from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.tuners_utils import BaseTunerLayer
import torch
from torch import nn

from rldx.configs.model.rldx import ActionModelLoraConfig, BackboneLoraConfig
from rldx.dist import rank_zero_print as _print


def inject_lora(
    module: nn.Module,
    lora: ActionModelLoraConfig | BackboneLoraConfig,
    target_modules: list[str],
    **peft_kwargs,
) -> None:
    config = LoraConfig(
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias="none",
        target_modules=target_modules,
        **peft_kwargs,
    )
    inject_adapter_in_model(config, module)
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    _print(f"[LoRA] {type(module).__name__} {lora} {peft_kwargs}: {trainable} / {total} trainable")


def to_base_layer_keys(model: nn.Module, state_dict: dict) -> dict:
    """Pre-LoRA checkpoints store ``X.weight``; PEFT moved it to ``X.base_layer.weight``."""
    wrapped = {
        key.replace(".base_layer.", "."): key for key in model.state_dict() if ".base_layer." in key
    }
    return {wrapped.get(key, key): value for key, value in state_dict.items()}


def fold_lora(model: nn.Module) -> None:
    """Merge each LoRA adapter into its base Linear and put the Linear back in its place."""
    adapted = [(name, m) for name, m in model.named_modules() if isinstance(m, BaseTunerLayer)]
    with torch.no_grad():
        for name, layer in adapted:
            base = layer.get_base_layer()
            down = layer.lora_A["default"].weight.float()
            up = layer.lora_B["default"].weight.float()
            base.weight.copy_(base.weight.float() + layer.scaling["default"] * up @ down)
            owner, _, attr = name.rpartition(".")
            setattr(model.get_submodule(owner), attr, base)
