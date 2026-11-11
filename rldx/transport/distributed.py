"""Tensor transport across distributed ranks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import math
import socket
from typing import Any, TypeVar

import torch
import torch.distributed as dist

from rldx.planner.routing import (
    BATCH_ROUTING_DESCRIPTOR_KEY,
    Assignment,
    BatchWork,
    Predictor,
    RoutePlan,
    Slot,
    make_route_plan,
)


ParsedBatch = TypeVar("ParsedBatch")

MAX_TENSOR_NDIM = 8
WIRE_ALIGNMENT = 16
MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_SLOT_BYTES = 8 * 1024 * 1024 * 1024
MAX_EXCHANGE_BYTES = 8 * 1024 * 1024 * 1024
_DTYPE_CODES = {
    torch.bool: 1,
    torch.uint8: 2,
    torch.int8: 3,
    torch.int16: 4,
    torch.int32: 5,
    torch.int64: 6,
    torch.float16: 7,
    torch.bfloat16: 8,
    torch.float32: 9,
    torch.float64: 10,
}
_CODE_DTYPES = {code: dtype for dtype, code in _DTYPE_CODES.items()}


class DistributedWireError(RuntimeError):
    """Base class for failures in the batch-routing wire contract."""


class DistributedWireCapacityError(DistributedWireError):
    """A valid frame cannot fit within the configured transport budget."""


class DistributedWireProtocolError(DistributedWireError):
    """A peer supplied an internally inconsistent wire frame."""


@dataclass(frozen=True)
class TensorHeader:
    numel: int
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> TensorHeader:
        if tensor.ndim > MAX_TENSOR_NDIM:
            raise RuntimeError("batch routing tensor rank exceeds wire header capacity")
        if tensor.dtype not in _DTYPE_CODES:
            raise TypeError(f"batch routing unsupported tensor dtype: {tensor.dtype}")
        return cls(
            tensor.numel(),
            tensor.numel() * tensor.element_size(),
            tensor.dtype,
            tuple(tensor.shape),
        )

    @classmethod
    def from_wire(cls, values: list[int]) -> TensorHeader:
        if not isinstance(values, list) or len(values) != 4 + MAX_TENSOR_NDIM:
            raise DistributedWireProtocolError("invalid tensor header width")
        if any(type(value) is not int for value in values):
            raise DistributedWireProtocolError("invalid tensor header field type")
        ndim, numel, nbytes, dtype_code = values[:4]
        if not 0 <= ndim <= MAX_TENSOR_NDIM:
            raise DistributedWireProtocolError("invalid tensor header ndim")
        dtype = _CODE_DTYPES.get(dtype_code)
        if dtype is None:
            raise DistributedWireProtocolError("invalid tensor header dtype code")
        shape = tuple(values[4 : 4 + ndim])
        if numel < 0 or nbytes < 0 or any(dimension < 0 for dimension in shape):
            raise DistributedWireProtocolError("invalid tensor header negative extent")
        if math.prod(shape) != numel:
            raise DistributedWireProtocolError("invalid tensor header shape/numel relation")
        element_size = torch.empty((), dtype=dtype).element_size()
        if numel * element_size != nbytes:
            raise DistributedWireProtocolError("invalid tensor header dtype/byte relation")
        return cls(numel, nbytes, dtype, shape)

    def to_wire(self) -> list[int]:
        ndim = len(self.shape)
        return (
            [ndim, self.numel, self.nbytes, _DTYPE_CODES[self.dtype]]
            + list(self.shape)
            + [0] * (MAX_TENSOR_NDIM - ndim)
        )


@dataclass
class _ExchangeBuffers:
    send: torch.Tensor
    receive: torch.Tensor


@dataclass(frozen=True)
class SlotMetadata:
    spec: Any
    tensor_headers: tuple[TensorHeader | None, ...]
    scalar_values: tuple[Any, ...]

    @classmethod
    def from_wire(cls, wire: bytes, entry: SlotEntry | None = None) -> SlotMetadata:
        try:
            raw = json.loads(wire)
            if not isinstance(raw, dict) or set(raw) != {"spec", "headers", "scalars"}:
                raise ValueError("fields")
            spec = _spec_from_wire(raw["spec"])
            headers = tuple(
                TensorHeader.from_wire(header) if header is not None else None
                for header in raw["headers"]
            )
            scalars = tuple(raw["scalars"])
            kinds = _spec_leaf_kinds(spec)
            if len(headers) != len(kinds) or len(scalars) != len(kinds):
                raise ValueError("cardinality")
            if any(
                (kind == "tensor" and (header is None or scalar is not None))
                or (kind == "scalar" and header is not None)
                for kind, header, scalar in zip(kinds, headers, scalars)
            ):
                raise ValueError("leaf kind")
            parsed = cls(spec, headers, scalars)
            if entry is not None:
                cursor = _align_wire_offset(entry.metadata_nbytes)
                for header in headers:
                    if header is not None:
                        cursor = _align_wire_offset(cursor + header.nbytes)
                if cursor != entry.nbytes:
                    raise ValueError("frame boundary")
            return parsed
        except DistributedWireProtocolError as error:
            raise DistributedWireProtocolError("invalid slot metadata") from error
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise DistributedWireProtocolError("invalid slot metadata") from error


@dataclass(frozen=True)
class SlotFrame:
    """One local batch framed for the wire: schema bytes first, packed tensors after."""

    spec: Any
    leaves: list[Any]
    metadata: SlotMetadata
    metadata_bytes: bytes
    nbytes: int


@dataclass(frozen=True)
class SlotEntry:
    """The fixed-width facts every rank learns about a slot before the exchange."""

    score: int
    metadata_nbytes: int
    nbytes: int

    @classmethod
    def from_wire(cls, values: list[int]) -> SlotEntry:
        if len(values) != 3 or any(type(value) is not int for value in values):
            raise DistributedWireProtocolError("invalid slot table entry")
        entry = cls(*values)
        if entry.metadata_nbytes < 0 or entry.nbytes < 0:
            raise DistributedWireProtocolError("invalid slot table entry")
        if entry.metadata_nbytes > entry.nbytes:
            raise DistributedWireProtocolError("invalid slot table entry")
        return entry


SlotTable = list[list[SlotEntry]]


def _host_rank_groups(hostnames: list[str]) -> list[list[int]]:
    """Group global ranks by their worker-pod hostname, preserving discovery order."""
    groups: dict[str, list[int]] = {}
    for rank, hostname in enumerate(hostnames):
        if not hostname:
            raise RuntimeError("distributed host discovery returned an empty hostname")
        groups.setdefault(hostname, []).append(rank)
    return list(groups.values())


def create_host_local_process_group():
    """Create the ProcessGroup containing ranks launched in this worker pod."""
    hostnames: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(hostnames, socket.gethostname())
    rank_groups = _host_rank_groups([hostname or "" for hostname in hostnames])
    return dist.new_subgroups_by_enumeration(
        rank_groups,
        backend=dist.get_backend(),
        group_desc="batch-routing/host-local",
    )


@dataclass
class ExchangeChannel:
    """Device transport owned for the life of one routing process group."""

    group: Any
    device: torch.device
    max_metadata_bytes: int = MAX_METADATA_BYTES
    max_slot_bytes: int = MAX_SLOT_BYTES
    max_exchange_bytes: int = MAX_EXCHANGE_BYTES
    stream: torch.cuda.Stream | None = field(init=False)
    staging: _ExchangeBuffers = field(init=False)

    def __post_init__(self) -> None:
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self.staging = _ExchangeBuffers(self._empty(0), self._empty(0))

    @classmethod
    def host_local(cls, device: torch.device) -> ExchangeChannel:
        group, _ = create_host_local_process_group()
        return cls(group, device)

    def staging_for(self, send_bytes: int, receive_bytes: int) -> _ExchangeBuffers:
        if self.staging.send.numel() < send_bytes:
            self.staging.send = self._empty(send_bytes)
        if self.staging.receive.numel() < receive_bytes:
            self.staging.receive = self._empty(receive_bytes)
        return self.staging

    def _empty(self, nbytes: int) -> torch.Tensor:
        return torch.empty(nbytes, dtype=torch.uint8, device=self.device)


def _align_wire_offset(offset: int) -> int:
    return (offset + WIRE_ALIGNMENT - 1) // WIRE_ALIGNMENT * WIRE_ALIGNMENT


def _record_tensors_on_stream(value: Any, stream: torch.cuda.Stream) -> None:
    """Tie every CUDA tensor in a routed pytree to the routing stream."""
    if torch.is_tensor(value):
        if value.device.type == "cuda":
            value.record_stream(stream)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _record_tensors_on_stream(item, stream)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _record_tensors_on_stream(item, stream)


def _flatten(value: Any, path: str = "batch") -> tuple[Any, list[Any]]:
    if torch.is_tensor(value):
        return ("tensor",), [value]
    if isinstance(value, Mapping):
        items, leaves = [], []
        for key in sorted(value):
            child, child_leaves = _flatten(value[key], f"{path}.{key}")
            items.append((key, child))
            leaves.extend(child_leaves)
        return ("mapping", tuple(items)), leaves
    if isinstance(value, tuple):
        children, leaves = [], []
        for index, item in enumerate(value):
            child, child_leaves = _flatten(item, f"{path}[{index}]")
            children.append(child)
            leaves.extend(child_leaves)
        return ("tuple", tuple(children)), leaves
    if isinstance(value, list):
        # Scalar lists such as harness image counts are one wire leaf.
        if all(item is None or isinstance(item, (bool, int, float, str)) for item in value):
            return ("scalar",), [value]
        children, leaves = [], []
        for index, item in enumerate(value):
            child, child_leaves = _flatten(item, f"{path}[{index}]")
            children.append(child)
            leaves.extend(child_leaves)
        return ("list", tuple(children)), leaves
    if value is None or isinstance(value, (bool, int, float, str)):
        return ("scalar",), [value]
    raise TypeError(f"batch routing unsupported pytree field at {path}: {type(value).__name__}")


def _unflatten(spec: Any, leaves: iter) -> Any:
    kind = spec[0]
    if kind in ("tensor", "scalar"):
        return next(leaves)
    if kind == "mapping":
        items = spec[1]
        values = {key: _unflatten(child, leaves) for key, child in items}
        return values
    if kind == "tuple":
        return tuple(_unflatten(child, leaves) for child in spec[1])
    if kind == "list":
        return [_unflatten(child, leaves) for child in spec[1]]
    raise AssertionError(kind)


def _spec_to_wire(spec: Any) -> Any:
    kind = spec[0]
    if kind in ("tensor", "scalar"):
        return [kind]
    if kind == "mapping":
        return [kind, [[key, _spec_to_wire(child)] for key, child in spec[1]]]
    if kind == "tuple":
        return [kind, [_spec_to_wire(child) for child in spec[1]]]
    if kind == "list":
        return [kind, [_spec_to_wire(child) for child in spec[1]]]
    raise AssertionError(kind)


def _spec_from_wire(spec: Any) -> Any:
    kind = spec[0]
    if kind in ("tensor", "scalar"):
        return (kind,)
    if kind == "mapping":
        return (
            kind,
            tuple((key, _spec_from_wire(child)) for key, child in spec[1]),
        )
    if kind == "tuple":
        return (kind, tuple(_spec_from_wire(child) for child in spec[1]))
    if kind == "list":
        return (kind, tuple(_spec_from_wire(child) for child in spec[1]))
    raise RuntimeError(f"batch routing received unknown schema node {kind!r}")


def _slot_metadata(spec: Any, leaves: list[Any]) -> SlotMetadata:
    return SlotMetadata(
        spec,
        tuple(
            TensorHeader.from_tensor(value) if torch.is_tensor(value) else None for value in leaves
        ),
        tuple(None if torch.is_tensor(value) else value for value in leaves),
    )


def _slot_metadata_to_wire(metadata: SlotMetadata) -> dict[str, Any]:
    return {
        "spec": _spec_to_wire(metadata.spec),
        "headers": [
            header.to_wire() if header is not None else None for header in metadata.tensor_headers
        ],
        "scalars": list(metadata.scalar_values),
    }


def _spec_leaf_kinds(spec: Any) -> list[str]:
    kind = spec[0]
    if kind in ("tensor", "scalar"):
        return [kind]
    if kind == "mapping":
        children = (child for _, child in spec[1])
    elif kind in ("tuple", "list"):
        children = iter(spec[1])
    else:
        raise DistributedWireProtocolError("invalid slot metadata schema")
    return [leaf for child in children for leaf in _spec_leaf_kinds(child)]


def _enforce_capacity(
    table: SlotTable,
    plan: RoutePlan,
    world: int,
    reuse_local: bool,
    *,
    max_metadata_bytes: int,
    max_slot_bytes: int,
    max_exchange_bytes: int,
) -> None:
    for entries in table:
        for entry in entries:
            if entry.metadata_nbytes > max_metadata_bytes:
                raise DistributedWireCapacityError("batch routing metadata capacity exceeded")
            if entry.nbytes > max_slot_bytes:
                raise DistributedWireCapacityError("batch routing slot capacity exceeded")
    for rank in range(world):
        send_bytes = sum(
            table[rank][item.source.micro].nbytes
            for item in plan
            if item.source.rank == rank and not (reuse_local and item.destination.rank == rank)
        )
        receive_bytes = sum(
            table[item.source.rank][item.source.micro].nbytes
            for item in plan
            if item.destination.rank == rank and not (reuse_local and item.source.rank == rank)
        )
        if max(send_bytes, receive_bytes) > max_exchange_bytes:
            raise DistributedWireCapacityError("batch routing exchange capacity exceeded")


def _wire_error_consensus(error: Exception | None, device: torch.device, group) -> None:
    if not dist.is_initialized() or dist.get_world_size(group) == 1:
        if error is not None:
            raise DistributedWireProtocolError(
                "batch routing wire validation failed on one or more ranks"
            ) from error
        return
    failed = torch.tensor([error is not None], dtype=torch.int32, device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
    if failed.item():
        raise DistributedWireProtocolError(
            "batch routing wire validation failed on one or more ranks"
        ) from error


def _frame(batch: Any) -> SlotFrame:
    spec, leaves = _flatten(batch)
    metadata = _slot_metadata(spec, leaves)
    metadata_bytes = json.dumps(_slot_metadata_to_wire(metadata), separators=(",", ":")).encode()
    nbytes = _align_wire_offset(len(metadata_bytes))
    for header in metadata.tensor_headers:
        if header is not None:
            nbytes = _align_wire_offset(nbytes + header.nbytes)
    return SlotFrame(spec, leaves, metadata, metadata_bytes, nbytes)


def _frames_device(frames: list[SlotFrame]) -> torch.device:
    devices = {value.device for frame in frames for value in frame.leaves if torch.is_tensor(value)}
    if len(devices) != 1:
        raise RuntimeError("batch routing batches must have tensor leaves on exactly one device")
    return next(iter(devices))


def _wire_policy(channel: ExchangeChannel, reuse_local: bool) -> tuple[int, ...]:
    return (
        channel.max_metadata_bytes,
        channel.max_slot_bytes,
        channel.max_exchange_bytes,
        int(reuse_local),
    )


def _gather_slot_table(
    entries: list[SlotEntry],
    device: torch.device,
    group,
    policy: tuple[int, ...],
) -> SlotTable:
    world = dist.get_world_size(group)
    local = torch.tensor(
        [
            *policy,
            *(
                value
                for entry in entries
                for value in (entry.score, entry.metadata_nbytes, entry.nbytes)
            ),
        ],
        dtype=torch.int64,
        device=device,
    )
    gathered = torch.empty(world * local.numel(), dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(gathered, local, group=group)
    rows = gathered.to("cpu").reshape(world, local.numel()).tolist()
    if any(tuple(row[: len(policy)]) != policy for row in rows):
        raise DistributedWireProtocolError("batch routing ranks use different wire policies")
    return [
        [
            SlotEntry.from_wire(row[offset : offset + 3])
            for offset in range(len(policy), len(row), 3)
        ]
        for row in rows
    ]


def _wire_order(
    plan: RoutePlan, source_rank: int, destination_rank: int, reuse_local: bool
) -> list[Assignment]:
    if reuse_local and source_rank == destination_rank:
        return []
    return sorted(
        (
            item
            for item in plan
            if item.source.rank == source_rank and item.destination.rank == destination_rank
        ),
        key=lambda item: item.source.micro,
    )


def _write_frame(payload: torch.Tensor, offset: int, frame: SlotFrame) -> int:
    metadata = torch.frombuffer(bytearray(frame.metadata_bytes), dtype=torch.uint8)
    payload[offset : offset + metadata.numel()].copy_(metadata)
    cursor = _align_wire_offset(offset + metadata.numel())
    for value in frame.leaves:
        if not torch.is_tensor(value):
            continue
        raw = value.contiguous().view(torch.uint8).reshape(-1)
        payload[cursor : cursor + raw.numel()].copy_(raw)
        cursor = _align_wire_offset(cursor + raw.numel())
    return cursor


def _read_leaves(payload: torch.Tensor, cursor: int, metadata: SlotMetadata) -> list[Any]:
    """Expose received tensors as contiguous views of the staging buffer."""
    leaves = []
    for leaf_index, header in enumerate(metadata.tensor_headers):
        if header is None:
            leaves.append(metadata.scalar_values[leaf_index])
            continue
        flat = payload[cursor : cursor + header.nbytes].view(header.dtype)
        if flat.numel() != header.numel:
            raise RuntimeError("batch routing tensor byte count does not match element count")
        leaves.append(flat.view(header.shape))
        cursor = _align_wire_offset(cursor + header.nbytes)
    return leaves


def _exchange(
    frames: list[SlotFrame],
    plan: RoutePlan,
    rank: int,
    world: int,
    channel: ExchangeChannel,
    table: SlotTable,
    reuse_local: bool,
    *,
    parse_batch: Callable[[Mapping[str, Any]], ParsedBatch],
    record_stream: torch.cuda.Stream | None = None,
) -> list[ParsedBatch]:
    device = _frames_device(frames)
    _enforce_capacity(
        table,
        plan,
        world,
        reuse_local,
        max_metadata_bytes=channel.max_metadata_bytes,
        max_slot_bytes=channel.max_slot_bytes,
        max_exchange_bytes=channel.max_exchange_bytes,
    )
    payload_in = [
        sum(frames[item.source.micro].nbytes for item in _wire_order(plan, rank, dest, reuse_local))
        for dest in range(world)
    ]
    payload_out = [
        sum(
            table[source][item.source.micro].nbytes
            for item in _wire_order(plan, source, rank, reuse_local)
        )
        for source in range(world)
    ]
    buffers = channel.staging_for(sum(payload_in), sum(payload_out))
    send_payload = buffers.send[: sum(payload_in)]
    receive_payload = buffers.receive[: sum(payload_out)]
    offset = 0
    for destination in range(world):
        for item in _wire_order(plan, rank, destination, reuse_local):
            offset = _write_frame(send_payload, offset, frames[item.source.micro])
    dist.all_to_all_single(
        receive_payload,
        send_payload,
        output_split_sizes=payload_out,
        input_split_sizes=payload_in,
        group=channel.group,
    )
    arrivals = []
    offset = 0
    for source in range(world):
        for item in _wire_order(plan, source, rank, reuse_local):
            entry = table[source][item.source.micro]
            prefix = torch.empty(
                entry.metadata_nbytes, dtype=torch.uint8, pin_memory=device.type == "cuda"
            )
            prefix.copy_(
                receive_payload[offset : offset + entry.metadata_nbytes], non_blocking=True
            )
            arrivals.append((item, offset, prefix))
            offset += entry.nbytes
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()
    result: dict[Slot, ParsedBatch] = {}
    wire_error = None
    try:
        for item, offset, prefix in arrivals:
            entry = table[item.source.rank][item.source.micro]
            metadata = SlotMetadata.from_wire(prefix.numpy().tobytes(), entry)
            cursor = _align_wire_offset(offset + prefix.numel())
            raw = _unflatten(metadata.spec, iter(_read_leaves(receive_payload, cursor, metadata)))
            result[item.destination] = parse_batch(raw)
        for item in plan:
            if reuse_local and item.source.rank == rank == item.destination.rank:
                frame = frames[item.source.micro]
                result[item.destination] = parse_batch(_unflatten(frame.spec, iter(frame.leaves)))
    except Exception as error:
        wire_error = error
    # Parsing local and received batches is part of the same collective boundary.
    _wire_error_consensus(wire_error, device, channel.group)
    routed = [result[Slot(rank, micro)] for micro in range(plan.group_size)]
    if record_stream is not None:
        for frame in frames:
            _record_tensors_on_stream(frame.leaves, record_stream)
        for tensor in (send_payload, receive_payload):
            tensor.record_stream(record_stream)
        for batch in routed:
            _record_tensors_on_stream(batch, record_stream)
    return routed


def exchange_batches(
    batches: list[Any],
    plan: RoutePlan,
    channel: ExchangeChannel,
    *,
    parse_batch: Callable[[Mapping[str, Any]], ParsedBatch],
    reuse_local: bool = True,
    record_stream: torch.cuda.Stream | None = None,
) -> list[ParsedBatch]:
    rank = dist.get_rank(channel.group)
    world = dist.get_world_size(channel.group)
    frames = None
    device = None
    error = None
    try:
        if len(batches) != plan.group_size:
            raise ValueError(
                f"batch routing plan expects {plan.group_size} local batches, got {len(batches)}"
            )
        frames = [_frame(batch) for batch in batches]
        device = _frames_device(frames)
    except Exception as caught:
        error = caught
    _wire_error_consensus(error, channel.device, channel.group)
    assert frames is not None and device is not None
    table = _gather_slot_table(
        [SlotEntry(0, len(frame.metadata_bytes), frame.nbytes) for frame in frames],
        device,
        channel.group,
        _wire_policy(channel, reuse_local),
    )
    return _exchange(
        frames,
        plan,
        rank,
        world,
        channel,
        table,
        reuse_local,
        parse_batch=parse_batch,
        record_stream=record_stream,
    )


def route_batches(
    batches: list[Any],
    channel: ExchangeChannel,
    *,
    predictor: Predictor,
    parse_batch: Callable[[Mapping[str, Any]], ParsedBatch],
) -> list[ParsedBatch]:
    entries = None
    frames = None
    device = None
    error = None
    try:
        if not batches:
            raise ValueError("batch routing requires a non-empty batch group")
        features = [
            BatchWork.parse(batch.pop(BATCH_ROUTING_DESCRIPTOR_KEY, None)) for batch in batches
        ]
        frames = [_frame(batch) for batch in batches]
        device = _frames_device(frames)
        if device != channel.device:
            raise RuntimeError(f"batch routing batches must live on {channel.device}, got {device}")
        entries = [
            SlotEntry(predictor.fixed_score(work), len(frame.metadata_bytes), frame.nbytes)
            for work, frame in zip(features, frames)
        ]
    except Exception as caught:
        error = caught
    _wire_error_consensus(error, channel.device, channel.group)
    assert entries is not None and frames is not None and device is not None
    table = _gather_slot_table(
        entries,
        device,
        channel.group,
        _wire_policy(channel, reuse_local=True),
    )
    plan = make_route_plan([[entry.score for entry in rank] for rank in table])
    rank = dist.get_rank(channel.group)
    world = dist.get_world_size(channel.group)
    if channel.stream is None:
        return _exchange(
            frames, plan, rank, world, channel, table, reuse_local=True, parse_batch=parse_batch
        )
    current = torch.cuda.current_stream(device)
    channel.stream.wait_stream(current)
    with torch.cuda.stream(channel.stream):
        for batch in batches:
            _record_tensors_on_stream(batch, channel.stream)
        result = _exchange(
            frames,
            plan,
            rank,
            world,
            channel,
            table,
            reuse_local=True,
            parse_batch=parse_batch,
            record_stream=channel.stream,
        )
    current.wait_stream(channel.stream)
    for batch in result:
        _record_tensors_on_stream(batch, current)
    return result
