"""
Distributed training utilities for multi-GPU training with DDP.

Usage:
    # Single GPU
    python -m monoart.motion.scripts.train --config ...

    # Multi-GPU with torchrun
    torchrun --nproc-per-node=2 -m monoart.motion.scripts.train --config ...

    # Multi-node
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr=xxx \
        -m monoart.motion.scripts.train --config ...
"""

import os

import torch
import torch.distributed as dist


def is_dist_available_and_initialized() -> bool:
    """Check if distributed training is available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """Get the number of processes in the distributed group."""
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get the rank of the current process."""
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """Get the local rank (GPU index on this node)."""
    if not is_dist_available_and_initialized():
        return 0
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed(backend: str = "nccl") -> bool:
    """
    Initialize distributed training.

    Args:
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)

    Returns:
        True if distributed training is initialized, False otherwise
    """
    # Check if launched with torchrun
    if "RANK" not in os.environ:
        print("Not launched with torchrun, running in single-GPU mode")
        return False

    # Get distributed info from environment
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Set the device for this process
    torch.cuda.set_device(local_rank)

    # Initialize the process group
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    # Synchronize all processes
    dist.barrier()

    if is_main_process():
        print("Distributed training initialized:")
        print(f"  World size: {world_size}")
        print(f"  Backend: {backend}")

    return True


def cleanup_distributed():
    """Clean up distributed training."""
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """
    Reduce a dictionary of tensors across all processes.

    Args:
        input_dict: Dictionary with tensor values
        average: If True, average the values; otherwise sum them

    Returns:
        Reduced dictionary
    """
    if not is_dist_available_and_initialized():
        return input_dict

    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []
        for k, v in sorted(input_dict.items()):
            names.append(k)
            if isinstance(v, torch.Tensor):
                values.append(v.clone())
            else:
                values.append(torch.tensor(v, device="cuda"))

        # Stack and reduce
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)

        if average:
            values /= world_size

        reduced_dict = {k: v.item() for k, v in zip(names, values)}

    return reduced_dict


def broadcast_object(obj, src: int = 0):
    """
    Broadcast an object from source rank to all other ranks.

    Args:
        obj: Object to broadcast (only used on src rank)
        src: Source rank

    Returns:
        Broadcasted object
    """
    if not is_dist_available_and_initialized():
        return obj

    if get_rank() == src:
        obj_list = [obj]
    else:
        obj_list = [None]

    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


class DistributedSamplerWrapper:
    """
    Wrapper to handle DistributedSampler for DataLoader.

    Ensures proper shuffling and epoch setting for distributed training.
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling."""
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
