# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed tensor utilities for efficient weight transfer."""

import math
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch.multiprocessing.reductions import reduce_tensor

# Default values for packed tensor configuration.
# These are imported by HCCLWeightTransferUpdateInfo and trainer_send_weights.
DEFAULT_PACKED_BUFFER_SIZE_BYTES = 1024 * 1024 * 1024  # 1GB
DEFAULT_PACKED_NUM_BUFFERS = 2


def packed_broadcast_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    group: Any,
    src: int,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
) -> None:
    """Broadcast tensors in a packed manner from trainer to workers.

    Packing a chunk, broadcasting it and releasing its buffer all happen on the
    caller's stream, and the broadcast is told to use that stream explicitly, so
    the barrier taken before a buffer is reused covers the collective that read
    it.

    Args:
        iterator: Iterator of model parameters. Returns a tuple of (name, tensor)
        group: Process group (PyHcclCommunicator)
        src: Source rank (0 in current implementation)
        post_iter_func: Function to apply to each (name, tensor) pair before
                       packing, should return a tensor
        buffer_size_bytes: Size in bytes for each packed tensor buffer.
                          Both producer and consumer must use the same value.
        num_buffers: Number of buffers for double/triple buffering.
                    Both producer and consumer must use the same value.
    """
    target_packed_tensor_size = buffer_size_bytes

    # Keep allocation, packing and broadcast on the caller's stream: the source
    # tensors may still be in flight on it, and the packed buffers are released
    # (and reused) on it, so a single stream is what orders the two.
    send_stream = torch.npu.current_stream()
    buffer_idx = 0
    packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]

    done = False
    while not done:
        # Finish the previous chunk's broadcast before reusing its buffer.
        send_stream.synchronize()
        with torch.npu.stream(send_stream):
            # Pack the tensors
            packing_tensor_list: list[torch.Tensor] = []
            packed_size = 0
            while True:
                try:
                    item = next(iterator)
                except StopIteration:
                    done = True
                    break
                # Apply post processing and convert to linearized uint8 tensor
                tensor = post_iter_func(item).contiguous().view(torch.uint8).view(-1)
                packing_tensor_list.append(tensor)
                packed_size += tensor.numel()
                if packed_size > target_packed_tensor_size:
                    break
            if not packing_tensor_list:
                # No more tensors — nothing left to broadcast
                break
            packed_tensors[buffer_idx] = torch.cat(packing_tensor_list, dim=0)
        # ``stream=`` is passed explicitly because ``stream=None`` resolves to
        # ``vllm_ascend.utils.current_stream()``, this process's cached stream,
        # rather than to the stream the packing ran on.
        group.broadcast(packed_tensors[buffer_idx], src=src, stream=send_stream)
        send_stream.synchronize()

        # Move to the next buffer
        buffer_idx = (buffer_idx + 1) % num_buffers

    # Drain before returning: the broadcasts are only *enqueued*, and the caller
    # may free the source tensors as soon as we return. It also keeps NPU tensor
    # cleanup at exit from hanging.
    send_stream.synchronize()


def packed_broadcast_consumer(
    iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
    group: Any,
    src: int,
    post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
) -> None:
    """Consume packed tensors and unpack them into a list of tensors.

    The receive buffer is allocated, filled by the broadcast and read by
    ``post_unpack_func`` on the caller's stream, and the broadcast is told to use
    that stream explicitly. A single stream is what orders the layerwise reload
    across chunks: the reload retains the unpacked tensors of a chunk and runs a
    layer's weight loaders when a later chunk completes that layer, so those
    loaders - and the buffer reuse that follows them - are all ordered on one
    stream instead of racing across two.

    Args:
        iterator: Iterator of parameter metadata. Returns (name, (shape, dtype))
        group: Process group (PyHcclCommunicator)
        src: Source rank (0 in current implementation)
        post_unpack_func: Function to apply to each list of (name, tensor) after
                         unpacking
        buffer_size_bytes: Size in bytes for each packed tensor buffer.
                          Both producer and consumer must use the same value.
        num_buffers: Number of buffers for double/triple buffering.
                    Both producer and consumer must use the same value.
    """

    def unpack_tensor(
        packed_tensor: torch.Tensor,
        names: list[str],
        shapes: list[list[int]],
        dtypes: list[torch.dtype],
        tensor_sizes: list[int],
    ) -> list[tuple[str, torch.Tensor]]:
        """Unpack a packed uint8 tensor into a list of typed tensors."""
        unpacked_tensors = packed_tensor.split(tensor_sizes)
        unpacked_list = [
            (name, tensor.contiguous().view(dtype).view(*shape))
            for name, shape, dtype, tensor in zip(names, shapes, dtypes, unpacked_tensors)
        ]
        return unpacked_list

    target_packed_tensor_size = buffer_size_bytes

    # Keep allocation, broadcast and weight loading on the caller's stream. A
    # layer completed by a later chunk reads the earlier chunk's retained buffer
    # views, so all of it has to be ordered on one stream: the loads run after the
    # broadcasts that filled the buffers, and before the slots are reused.
    load_stream = torch.npu.current_stream()
    buffer_idx = 0

    packing_tensor_meta_data: list[list[tuple[str, list[int], torch.dtype, int]]] = [[] for _ in range(num_buffers)]
    packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]

    done = False
    while not done:
        # Finish the previous chunk's broadcast and weight loading before reusing
        # its buffer.
        load_stream.synchronize()
        with torch.npu.stream(load_stream):
            # Collect parameter metadata for this buffer
            packing_tensor_meta_data[buffer_idx] = []
            packed_size = 0
            while True:
                try:
                    name, (shape, dtype) = next(iterator)
                except StopIteration:
                    done = True
                    break
                tensor_size = math.prod(shape) * dtype.itemsize
                packing_tensor_meta_data[buffer_idx].append((name, shape, dtype, tensor_size))
                packed_size += tensor_size
                if packed_size > target_packed_tensor_size:
                    break
            if not packing_tensor_meta_data[buffer_idx]:
                break
            packed_tensors[buffer_idx] = torch.empty(
                packed_size,
                dtype=torch.uint8,
                device="npu",
            )
        # ``stream=`` is passed explicitly because ``stream=None`` resolves to
        # ``vllm_ascend.utils.current_stream()``, this process's cached stream,
        # rather than to the stream the buffer was allocated on.
        group.broadcast(packed_tensors[buffer_idx], src=src, stream=load_stream)
        # Finish receiving before the model's weight loader reads the buffer.
        load_stream.synchronize()
        with torch.npu.stream(load_stream):
            names, shapes, dtypes, tensor_sizes = zip(*packing_tensor_meta_data[buffer_idx])
            post_unpack_func(
                unpack_tensor(
                    packed_tensors[buffer_idx],
                    list(names),
                    list(shapes),
                    list(dtypes),
                    list(tensor_sizes),
                )
            )

        # Move to the next buffer
        buffer_idx = (buffer_idx + 1) % num_buffers

    # Drain before returning: the weight loading is only *enqueued* and the weights
    # are applied once the device has executed it. Otherwise NPU tensor cleanup at
    # exit may hang.
    load_stream.synchronize()


# ── NPU IPC packed transfer ────────────────────────────────────────────


def packed_npu_ipc_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    npu_uuid: str,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
) -> Iterator[dict[str, Any]]:
    """Pack tensors into a reusable NPU IPC buffer and yield chunks.

    Allocates a single NPU buffer of ``buffer_size_bytes`` and registers
    it for IPC once via ``reduce_tensor``.  Each chunk's packed data is
    copied into this buffer before yielding, so only one IPC-shared
    allocation exists for the lifetime of the transfer.

    Args:
        iterator: Iterator of (name, tensor) pairs.
        npu_uuid: Physical NPU UUID string for this rank.
        post_iter_func: Applied to each (name, tensor) before packing.
        buffer_size_bytes: Exact capacity of the reusable IPC buffer.
    """
    ipc_buffer = torch.empty(buffer_size_bytes, dtype=torch.uint8, device="npu")
    # Store only the rebuild args (drop the func); the consumer rebuilds with
    # the well-known ``rebuild_npu_tensor``, mirroring upstream's CUDA IPC engine.
    _, ipc_args = reduce_tensor(ipc_buffer)

    names: list[str] = []
    shapes: list[list[int]] = []
    dtypes: list[torch.dtype] = []
    tensor_sizes: list[int] = []
    total_bytes = 0

    for name, orig_tensor in iterator:
        flat = post_iter_func((name, orig_tensor)).contiguous().view(torch.uint8).view(-1)

        if flat.numel() > buffer_size_bytes:
            raise ValueError(
                f"Tensor '{name}' has size {flat.numel()} bytes, "
                f"which exceeds buffer_size_bytes={buffer_size_bytes}. "
                f"Increase buffer_size_bytes to at least {flat.numel()}."
            )

        if total_bytes and total_bytes + flat.numel() > buffer_size_bytes:
            torch.npu.current_stream().synchronize()
            yield {
                "names": names,
                "shapes": shapes,
                "dtype_names": [str(d).split(".")[-1] for d in dtypes],
                "tensor_sizes": tensor_sizes,
                "ipc_handle": {npu_uuid: ipc_args},
            }
            names, shapes, dtypes, tensor_sizes = [], [], [], []
            total_bytes = 0

        ipc_buffer[total_bytes : total_bytes + flat.numel()].copy_(flat)
        names.append(name)
        shapes.append(list(orig_tensor.shape))
        dtypes.append(orig_tensor.dtype)
        tensor_sizes.append(flat.numel())
        total_bytes += flat.numel()

    if total_bytes:
        torch.npu.current_stream().synchronize()
        yield {
            "names": names,
            "shapes": shapes,
            "dtype_names": [str(d).split(".")[-1] for d in dtypes],
            "tensor_sizes": tensor_sizes,
            "ipc_handle": {npu_uuid: ipc_args},
        }


def packed_npu_ipc_consumer(
    ipc_handle: dict[str, tuple],
    physical_npu_id: str,
    names: list[str],
    shapes: list[list[int]],
    dtype_names: list[str],
    tensor_sizes: list[int],
    device_index: int,
) -> list[tuple[str, torch.Tensor]]:
    """Unpack a single packed IPC chunk into named tensors.

    Reconstructs the packed buffer via the IPC handle, unpacks into
    individual tensors, and clones each into independent storage before
    returning.  The clone is required because the producer reuses one
    IPC buffer across chunks.

    Args:
        ipc_handle: Mapping of NPU UUID to a ``rebuild_npu_tensor`` args tuple
            from ``reduce_tensor``.
        physical_npu_id: Physical NPU UUID string for the current process.
        names: Parameter names in the packed buffer.
        shapes: Parameter shapes.
        dtype_names: Parameter dtype name strings (e.g. "float16").
        tensor_sizes: Size in bytes of each parameter in the packed buffer.
        device_index: Local NPU device index.
    """
    # Lazy import: ``rebuild_npu_tensor`` lives in ``torch_npu`` and must not be
    # imported at module load time on non-NPU hosts.
    from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

    if physical_npu_id not in ipc_handle:
        raise ValueError(
            f"IPC handle not found for NPU UUID {physical_npu_id}. Available UUIDs: {list(ipc_handle.keys())}"
        )

    args = ipc_handle[physical_npu_id]
    list_args = list(args)
    # Index 6 of the args from reduce_tensor is the device_index.
    # Overwrite it with the receiver's device index.
    list_args[6] = device_index
    packed = rebuild_npu_tensor(*list_args)

    content_size = sum(tensor_sizes)
    packed = packed[:content_size]

    dtypes = [getattr(torch, dn) for dn in dtype_names]
    weights: list[tuple[str, torch.Tensor]] = []
    offset = 0
    for name, shape, dtype, size in zip(names, shapes, dtypes, tensor_sizes):
        raw = packed[offset : offset + size]
        tensor = raw.contiguous().view(dtype).view(*shape).clone()
        weights.append((name, tensor))
        offset += size

    return weights
