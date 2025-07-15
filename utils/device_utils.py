"""Device and hardware utilities."""

import torch
import gc
from typing import Optional, Union, Dict


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info() -> Dict[str, Union[str, int, float]]:
    """Get information about the current device."""
    device = get_device()
    info = {"device": str(device)}
    
    if device.type == "cuda":
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
            "gpu_memory_allocated": torch.cuda.memory_allocated(0),
            "gpu_memory_reserved": torch.cuda.memory_reserved(0),
        })
    
    return info


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def move_to_device(data: Union[torch.Tensor, Dict, list], device: Optional[torch.device] = None):
    """Recursively move data to device."""
    if device is None:
        device = get_device()
    
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    else:
        return data


def estimate_model_memory(model: torch.nn.Module) -> Dict[str, int]:
    """Estimate memory usage of a model."""
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
    
    return {
        "param_memory_bytes": param_memory,
        "buffer_memory_bytes": buffer_memory,
        "total_memory_bytes": param_memory + buffer_memory,
        "param_memory_mb": param_memory / (1024 * 1024),
        "total_memory_mb": (param_memory + buffer_memory) / (1024 * 1024),
    }


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
