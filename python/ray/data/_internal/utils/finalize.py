import threading
from typing import Any, Optional, Union

import torch

from ray.data.collate_fn import (
    TensorBatchReturnType,
    TensorBatchType,
    is_tensor_batch_type,
)
from ray.data.util.torch_utils import (
    DEFAULT_TENSOR_NON_BLOCKING_TRANSFER,
    move_tensors_to_device,
)

CustomBatchType = Any


class DefaultFinalizeFn:
    """finalize_fn that overlaps the host->device transfer with downstream GPU compute.

    time ──────────────────────────────────────────────────────────────▶

                        (1)                    (3)
                         ▼                      ▼
               ┌─────────┐┌─────────┐┌─────────┐
    transfer   │ load A  ││ load B  ││ load C  │ ─────▶  each "load" = pinned
    stream     └─────────┘└─────────┘└─────────┘         H2D copy on the
                    │                                    transfer stream
                (2) │
                    ▼
                          ┌─────────┐┌─────────┐
    compute               │compute A││compute B│ ─────▶   compute runs on the
    stream                └─────────┘└─────────┘          default stream
                                                            ▲
                          └── overlap ──┘  load B copies while compute A runs

    phase 1        phase 2                 phase 3
    (load A only)  (compute A ‖ load B)    (compute B ‖ load C)

    ─────────────────────────────────────────────────────────────────────
    (1) wait_event: the compute stream waits until batch A is FULLY
        transferred before it starts computing on A.
    (2) record_stream: the compute stream marks batch A as "in use", so the
        transfer stream will not reuse / overwrite A's buffer until compute
        has finished with it.
    (3) once compute on A completes, its buffer is freed and can be reused
        by a later load.

    Preconditions for real overlap:
    - The collated source tensors are pinned (e.g. ``pin_memory=True``);
      otherwise the non-blocking H2D copy from pageable memory is effectively
      synchronous and nothing overlaps.

    Important Notes:
    - The consumer MUST compute on the passed ``compute_stream``. This is important for
      correct ordering of operations (points 1, 2, 3 in diagram).
    """

    def __init__(
        self,
        device: "torch.device",
        compute_stream: Optional["torch.cuda.Stream"],
    ):
        """Construct the DefaultFinalizeFn.

        Args:
            device: The CUDA device to transfer tensor batches to.
            compute_stream: The CUDA stream that will run computation on the
                transferred batches. Note that when device is a cuda device,
                compute_stream must be specified.
        """
        self._device = device
        self._compute_stream = compute_stream

        # Lazily initialized: the transfer may not be needed (e.g. CPU device), and
        # the copy stream must be created on the finalize thread.
        self._copy_stream: Optional["torch.cuda.Stream"] = None
        self._init_lock = threading.Lock()

    @torch.no_grad()
    def __call__(
        self, batch: Union[TensorBatchType, CustomBatchType]
    ) -> Union[TensorBatchReturnType, CustomBatchType]:
        if not is_tensor_batch_type(batch):
            return batch

        # CPU target: no stream/event coordination needed. move_tensors_to_device
        # still handles the chunk concatenation and the shape dispatch.
        if not self._is_cuda():
            return move_tensors_to_device(
                batch,
                device=self._device,
                non_blocking=DEFAULT_TENSOR_NON_BLOCKING_TRANSFER,
            )

        # Initialize copy stream if needed.
        self._lazy_init()

        assert self._copy_stream is not None
        assert self._compute_stream is not None

        with torch.cuda.stream(self._copy_stream):
            moved = move_tensors_to_device(
                batch,
                device=self._device,
                non_blocking=DEFAULT_TENSOR_NON_BLOCKING_TRANSFER,
            )

        # The outputs were allocated on the copy stream but will be read on the
        # compute stream; tell the allocator so it doesn't recycle them early.
        _record_stream(moved, self._compute_stream)

        # Order the compute stream after copy is fully complete.
        copy_done = self._copy_stream.record_event()
        self._compute_stream.wait_event(copy_done)
        return moved

    def _is_cuda(self) -> bool:
        assert self._device is not None
        return self._device.type == "cuda"

    def _lazy_init(self) -> None:
        if self._copy_stream is not None:
            # Fast fail path without acquiring lock.
            return
        with self._init_lock:
            if self._copy_stream is None:
                self._copy_stream = torch.cuda.Stream(self._device)


def _record_stream(batch: TensorBatchReturnType, stream: "torch.cuda.Stream") -> None:
    """Recursively call ``record_stream(stream)`` on every tensor in ``batch``.

    ``move_tensors_to_device`` returns a tensor, a (list/tuple of) tensor(s), or a
    dict of tensors, so we recurse to cover all of them.
    """
    if isinstance(batch, torch.Tensor):
        batch.record_stream(stream)
    elif isinstance(batch, dict):
        for value in batch.values():
            _record_stream(value, stream)
    elif isinstance(batch, (list, tuple)):
        for value in batch:
            _record_stream(value, stream)
