from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F


def pad_rows(tensor: torch.Tensor, padding: int, chunks: int) -> torch.Tensor:
    return torch.cat(
        [
            F.pad(part, (0, 0) * (part.dim() - 1) + (0, padding))
            for part in tensor.chunk(chunks, dim=0)
        ]
    )


def logical_rows(tensor: torch.Tensor, logical: int, chunks: int) -> torch.Tensor:
    return torch.cat([part[:logical] for part in tensor.chunk(chunks, dim=0)])


def _pad_rows_on_load(module, state_dict, prefix, *_, logical: int, chunks: int) -> None:
    padding = module.out_features // chunks - logical
    for key in (f"{prefix}weight", f"{prefix}bias"):
        value = state_dict.get(key)
        if value is not None and value.shape[0] == chunks * logical:
            state_dict[key] = pad_rows(value, padding, chunks)


def _pad_columns_on_load(module, state_dict, prefix, *_, logical: int) -> None:
    key = f"{prefix}weight"
    value = state_dict.get(key)
    if value is not None and value.shape[1] == logical:
        state_dict[key] = F.pad(value, (0, module.in_features - logical))


def _canonical_state_dict(
    module,
    state_dict,
    prefix,
    _local_metadata,
    *,
    rows: str,
    columns: str,
    chunks: int,
    logical: int,
) -> None:
    for key in (f"{prefix}{rows}.weight", f"{prefix}{rows}.bias"):
        if key in state_dict:
            state_dict[key] = logical_rows(state_dict[key], logical, chunks)
    key = f"{prefix}{columns}.weight"
    state_dict[key] = state_dict[key][:, :logical]


def _pad_state_dict(
    module, state_dict, prefix, *_, rows: str, columns: str, chunks: int, logical: int
) -> None:
    _pad_rows_on_load(
        getattr(module, rows), state_dict, f"{prefix}{rows}.", logical=logical, chunks=chunks
    )
    _pad_columns_on_load(
        getattr(module, columns), state_dict, f"{prefix}{columns}.", logical=logical
    )


@dataclass(frozen=True)
class HiddenLayout:
    rows: str
    columns: str
    chunks: int
    logical: int


def pad_hidden(module: nn.Module, rows: str, columns: str, alignment: int, chunks: int = 1) -> None:
    layout = HiddenLayout(rows, columns, chunks, getattr(module, columns).in_features)
    module.hidden_layout = layout
    hooks = {"rows": rows, "columns": columns, "chunks": chunks, "logical": layout.logical}
    module.register_state_dict_post_hook(partial(_canonical_state_dict, **hooks))
    module.register_load_state_dict_pre_hook(partial(_pad_state_dict, **hooks))
    getattr(module, rows).register_load_state_dict_pre_hook(
        partial(_pad_rows_on_load, logical=layout.logical, chunks=chunks)
    )
    getattr(module, columns).register_load_state_dict_pre_hook(
        partial(_pad_columns_on_load, logical=layout.logical)
    )
    realign_hidden(module, alignment)


def realign_hidden_widths(model: nn.Module, alignment: int) -> None:
    for module in model.modules():
        if hasattr(module, "hidden_layout"):
            realign_hidden(module, alignment)


def realign_hidden(module: nn.Module, alignment: int) -> None:
    layout = module.hidden_layout
    first, second = getattr(module, layout.rows), getattr(module, layout.columns)
    runtime = -(-layout.logical // alignment) * alignment
    if second.in_features == runtime:
        return
    padding = runtime - layout.logical
    with torch.no_grad():
        first.weight = nn.Parameter(
            pad_rows(
                logical_rows(first.weight, layout.logical, layout.chunks), padding, layout.chunks
            ),
            requires_grad=first.weight.requires_grad,
        )
        if first.bias is not None:
            first.bias = nn.Parameter(
                pad_rows(
                    logical_rows(first.bias, layout.logical, layout.chunks), padding, layout.chunks
                ),
                requires_grad=first.bias.requires_grad,
            )
        second.weight = nn.Parameter(
            F.pad(second.weight[:, : layout.logical], (0, padding)),
            requires_grad=second.weight.requires_grad,
        )
    first.out_features = layout.chunks * runtime
    second.in_features = runtime
