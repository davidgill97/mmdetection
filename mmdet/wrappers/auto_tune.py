# Copyright (c) OpenMMLab. All rights reserved.
"""Auto-tuning utilities for batch size and learning rate optimization."""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, auto-tuning features disabled")


def get_gpu_info() -> Tuple[Optional[int], Optional[float]]:
    """Get GPU count and available memory.
    
    Returns:
        Tuple[Optional[int], Optional[float]]: (num_gpus, memory_mb_per_gpu)
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return None, None
    
    try:
        num_gpus = torch.cuda.device_count()
        # Get memory for the first GPU (assuming all GPUs are similar)
        mem_free, mem_total = torch.cuda.mem_get_info(0)
        memory_mb = mem_free / (1024 ** 2)
        return num_gpus, memory_mb
    except Exception as e:
        logger.warning(f"Failed to get GPU info: {e}")
        return None, None


def estimate_batch_size(
    model_size: str = 'medium',
    gpu_memory_mb: Optional[float] = None,
    image_size: Tuple[int, int] = (640, 640),
    conservative: bool = True
) -> int:
    """Estimate optimal batch size based on model and GPU constraints.
    
    Args:
        model_size (str): Model size category ('tiny', 'small', 'medium', 'large', 'xlarge')
        gpu_memory_mb (Optional[float]): Available GPU memory in MB
        image_size (Tuple[int, int]): Input image size (height, width)
        conservative (bool): Use conservative estimates to avoid OOM
        
    Returns:
        int: Recommended batch size per GPU
        
    Examples:
        >>> # Auto-detect GPU memory
        >>> batch_size = estimate_batch_size('medium')
        >>> 
        >>> # Specify GPU memory manually
        >>> batch_size = estimate_batch_size('large', gpu_memory_mb=24000)
    """
    if gpu_memory_mb is None:
        _, gpu_memory_mb = get_gpu_info()
        if gpu_memory_mb is None:
            logger.warning("Cannot determine GPU memory, using default batch size")
            return 2  # Safe default
    
    # Memory overhead factor (reduces effective memory for safety)
    overhead_factor = 0.7 if conservative else 0.85
    effective_memory = gpu_memory_mb * overhead_factor
    
    # Base memory estimates per sample (in MB) for different model sizes
    # These account for: forward pass, backward pass, optimizer states, gradients
    memory_per_sample = {
        'tiny': 200,      # e.g., YOLOX-Tiny, efficientdet-d0
        'small': 400,     # e.g., YOLOX-S, RetinaNet-R50
        'medium': 800,    # e.g., DETR-R50, Faster R-CNN-R50
        'large': 1600,    # e.g., DETR-R101, Cascade R-CNN
        'xlarge': 3200,   # e.g., Swin-L models
    }
    
    # Adjust for image size (relative to 640x640)
    base_pixels = 640 * 640
    actual_pixels = image_size[0] * image_size[1]
    size_factor = actual_pixels / base_pixels
    
    mem_per_sample = memory_per_sample.get(model_size, memory_per_sample['medium'])
    mem_per_sample *= size_factor
    
    # Calculate batch size
    batch_size = max(1, int(effective_memory / mem_per_sample))
    
    # Clip to reasonable range
    batch_size = min(batch_size, 64)  # Upper limit
    
    logger.info(
        f"Estimated batch size: {batch_size} "
        f"(GPU: {gpu_memory_mb:.0f}MB, Model: {model_size}, "
        f"Image: {image_size[0]}x{image_size[1]})"
    )
    
    return batch_size


def auto_scale_lr(
    base_lr: float,
    base_batch_size: int,
    new_batch_size: int,
    num_gpus: int = 1,
    scaling_rule: str = 'linear',
    warmup_iters: Optional[int] = None
) -> dict:
    """Automatically scale learning rate based on batch size and number of GPUs.
    
    The linear scaling rule (Goyal et al., 2017) states that when batch size
    is multiplied by k, the learning rate should also be multiplied by k.
    
    Args:
        base_lr (float): Base learning rate from config
        base_batch_size (int): Base batch size from config
        new_batch_size (int): New batch size per GPU
        num_gpus (int): Number of GPUs for distributed training
        scaling_rule (str): Scaling rule ('linear', 'sqrt', 'none')
        warmup_iters (Optional[int]): Number of warmup iterations (auto-calculated if None)
        
    Returns:
        dict: Dictionary with 'lr', 'warmup_iters', and 'warmup_ratio'
        
    References:
        - Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
        - https://arxiv.org/abs/1706.02677
        
    Examples:
        >>> # Linear scaling for 4 GPUs
        >>> result = auto_scale_lr(0.001, 8, 16, num_gpus=4)
        >>> print(result['lr'])  # 0.008 (8x scaling)
        
        >>> # With warmup
        >>> result = auto_scale_lr(0.001, 8, 32, num_gpus=8, warmup_iters=1000)
    """
    total_base_batch = base_batch_size
    total_new_batch = new_batch_size * num_gpus
    
    # Calculate scaling factor
    scale_factor = total_new_batch / total_base_batch
    
    # Apply scaling rule
    if scaling_rule == 'linear':
        scaled_lr = base_lr * scale_factor
    elif scaling_rule == 'sqrt':
        # Square root scaling (used in some papers for very large batch sizes)
        scaled_lr = base_lr * (scale_factor ** 0.5)
    elif scaling_rule == 'none':
        scaled_lr = base_lr
    else:
        raise ValueError(f"Unknown scaling rule: {scaling_rule}")
    
    # Calculate warmup iterations
    # Rule of thumb: use more warmup for larger learning rates
    if warmup_iters is None:
        # Default: 500 iterations for base case, scale proportionally
        warmup_iters = max(500, int(500 * scale_factor))
        # Cap at 5000 iterations
        warmup_iters = min(warmup_iters, 5000)
    
    # Calculate warmup ratio (warmup_iters / total_iters)
    # This is used by some schedulers
    warmup_ratio = min(0.1, warmup_iters / 10000)  # Assume ~100k total iters
    
    result = {
        'lr': scaled_lr,
        'warmup_iters': warmup_iters,
        'warmup_ratio': warmup_ratio,
        'scale_factor': scale_factor,
        'scaling_rule': scaling_rule
    }
    
    logger.info(
        f"LR scaling: {base_lr:.6f} -> {scaled_lr:.6f} "
        f"(factor: {scale_factor:.2f}x, rule: {scaling_rule}, "
        f"batch: {total_base_batch} -> {total_new_batch})"
    )
    logger.info(f"Warmup: {warmup_iters} iterations (ratio: {warmup_ratio:.3f})")
    
    return result


def suggest_training_config(
    config_path: str,
    num_gpus: int = 1,
    auto_batch: bool = True,
    auto_lr: bool = True,
    target_memory_usage: float = 0.7
) -> dict:
    """Suggest optimal training configuration based on hardware.
    
    Args:
        config_path (str): Path to model config
        num_gpus (int): Number of GPUs available
        auto_batch (bool): Enable automatic batch size tuning
        auto_lr (bool): Enable automatic learning rate scaling
        target_memory_usage (float): Target GPU memory usage (0.0-1.0)
        
    Returns:
        dict: Suggested configuration with 'batch_size', 'lr', 'amp', etc.
        
    Examples:
        >>> # Get suggestions for 4-GPU setup
        >>> suggestions = suggest_training_config(
        ...     'configs/detr/detr_r50.py',
        ...     num_gpus=4,
        ...     auto_batch=True,
        ...     auto_lr=True
        ... )
        >>> print(f"Suggested batch size: {suggestions['batch_size']}")
        >>> print(f"Suggested learning rate: {suggestions['lr']}")
    """
    from mmengine.config import Config
    
    # Load config
    cfg = Config.fromfile(config_path)
    
    # Get base values
    base_batch_size = getattr(cfg.train_dataloader, 'batch_size', 2)
    base_lr = getattr(cfg.optim_wrapper.optimizer, 'lr', 0.0001)
    
    suggestions = {
        'batch_size': base_batch_size,
        'lr': base_lr,
        'num_gpus': num_gpus,
        'amp': False,
        'warmup_iters': 500,
    }
    
    # Auto-tune batch size
    if auto_batch:
        num_devices, gpu_memory_mb = get_gpu_info()
        if gpu_memory_mb is not None:
            # Estimate model size from config name
            config_name = config_path.lower()
            if 'tiny' in config_name or 'nano' in config_name:
                model_size = 'tiny'
            elif 'small' in config_name:
                model_size = 'small'
            elif 'large' in config_name or 'x101' in config_name:
                model_size = 'large'
            elif 'xlarge' in config_name or 'swin' in config_name:
                model_size = 'xlarge'
            else:
                model_size = 'medium'
            
            suggestions['batch_size'] = estimate_batch_size(
                model_size=model_size,
                gpu_memory_mb=gpu_memory_mb,
                conservative=(target_memory_usage <= 0.75)
            )
    
    # Auto-scale learning rate
    if auto_lr and suggestions['batch_size'] != base_batch_size:
        lr_config = auto_scale_lr(
            base_lr=base_lr,
            base_batch_size=base_batch_size,
            new_batch_size=suggestions['batch_size'],
            num_gpus=num_gpus
        )
        suggestions['lr'] = lr_config['lr']
        suggestions['warmup_iters'] = lr_config['warmup_iters']
        suggestions['warmup_ratio'] = lr_config['warmup_ratio']
    
    # Suggest AMP for modern GPUs with sufficient memory
    _, gpu_memory_mb = get_gpu_info()
    if gpu_memory_mb is not None and gpu_memory_mb > 8000:
        suggestions['amp'] = True
        logger.info("Suggesting AMP (automatic mixed precision) for faster training")
    
    # Print summary
    logger.info("=" * 70)
    logger.info("Suggested Training Configuration:")
    logger.info(f"  Batch size per GPU: {suggestions['batch_size']}")
    logger.info(f"  Total batch size: {suggestions['batch_size'] * num_gpus}")
    logger.info(f"  Learning rate: {suggestions['lr']:.6f}")
    logger.info(f"  Warmup iterations: {suggestions.get('warmup_iters', 'N/A')}")
    logger.info(f"  AMP enabled: {suggestions['amp']}")
    logger.info(f"  Number of GPUs: {num_gpus}")
    logger.info("=" * 70)
    
    return suggestions
