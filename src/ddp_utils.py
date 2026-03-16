#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DDP (Distributed Data Parallel) Utilities

Provides functions for initializing and managing distributed training:
- init_ddp: Initialize DDP process group
- barrier: Synchronization barrier
- log0: Logging only from rank 0
"""

import os
import torch
import torch.distributed as dist
from typing import Tuple

from .log import get_logger

logger = get_logger(__name__)


def init_ddp(backend: str = "nccl") -> Tuple[bool, int, int, int, torch.device]:
    """
    Initialize Distributed Data Parallel environment.

    Detects if running under torchrun/DDP and initializes accordingly.
    Falls back to single-process mode if DDP environment not detected.

    Args:
        backend: Communication backend ("nccl" for GPU, "gloo" for CPU)

    Returns:
        Tuple of:
            - is_ddp: Whether DDP is enabled
            - rank: Global rank of this process
            - world_size: Total number of processes
            - local_rank: Local rank on this node
            - device: torch.device for this process

    Example:
        >>> is_ddp, rank, world, local, device = init_ddp("nccl")
        >>> print(f"DDP enabled: {is_ddp}, rank {rank}/{world}")
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return True, rank, world_size, local_rank, device

    # Single-process mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, 0, device


def barrier(is_ddp: bool) -> None:
    """
    Synchronization barrier for DDP processes.

    All processes must call this function; execution blocks until
    all processes have reached this point.

    Args:
        is_ddp: Whether DDP is enabled (from init_ddp)
    """
    if is_ddp and dist.is_initialized():
        dist.barrier()


def log0(is_ddp: bool, rank: int, msg: str) -> None:
    """
    Print message only from rank 0.

    In DDP mode, only the process with rank 0 prints the message.
    In single-process mode, always prints.

    Args:
        is_ddp: Whether DDP is enabled
        rank: Current process rank
        msg: Message to print

    Example:
        >>> log0(is_ddp, rank, "[Step 1] Training started")
    """
    if (not is_ddp) or rank == 0:
        logger.info(msg)


def cleanup_ddp() -> None:
    """
    Clean up DDP process group.

    Should be called at the end of training to properly tear down
    the distributed environment.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
