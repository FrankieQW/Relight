#!/usr/bin/env python3
"""Minimal two-rank NCCL smoke test, independent of the training model."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    value = torch.tensor([float(dist.get_rank() + 1)], device=f"cuda:{local_rank}")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(local_rank)
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"NCCL all_reduce mismatch: got {value.item()}, expected {expected}")
    print(
        f"rank={dist.get_rank()} device={local_rank} "
        f"gpu={torch.cuda.get_device_name(local_rank)} all_reduce={value.item()} OK",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
