#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Simplified training script with automatic configuration tuning.

This script provides an easy-to-use interface for training with automatic
batch size and learning rate tuning based on available GPU resources.

Example usage:
    # Simplest: just provide config
    python tools/train_auto.py configs/detr/detr_r50_8xb2-150e_coco.py
    
    # With automatic batch size tuning
    python tools/train_auto.py configs/detr/detr_r50_8xb2-150e_coco.py --auto-batch
    
    # Multi-GPU with automatic LR scaling
    python -m torch.distributed.launch --nproc_per_node=4 \
        tools/train_auto.py configs/detr/detr_r50_8xb2-150e_coco.py --auto-scale-lr
    
    # All automatic features
    python tools/train_auto.py configs/detr/detr_r50_8xb2-150e_coco.py \
        --auto-batch --auto-scale-lr --amp
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmdet.wrappers import HydraWrapper
from mmdet.utils import setup_cache_size_limit_of_dynamo

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import wandb
    wandb.require("core")
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available or failed to initialize")


def get_gpu_memory_mb():
    """Get available GPU memory in MB."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return None
    
    try:
        # Get memory for the first GPU
        mem_free, mem_total = torch.cuda.mem_get_info(0)
        return mem_free / (1024 ** 2)  # Convert to MB
    except Exception as e:
        logger.warning(f"Failed to get GPU memory: {e}")
        return None


def estimate_optimal_batch_size(config, gpu_memory_mb=None, num_gpus=1):
    """Estimate optimal batch size based on GPU memory and model.
    
    Args:
        config: Model configuration
        gpu_memory_mb (float): Available GPU memory in MB
        num_gpus (int): Number of GPUs
        
    Returns:
        int: Recommended batch size per GPU
    """
    if gpu_memory_mb is None:
        gpu_memory_mb = get_gpu_memory_mb()
        if gpu_memory_mb is None:
            logger.warning("Cannot determine GPU memory, using default batch size")
            return None
    
    # Conservative estimates for different memory sizes
    # These are rough heuristics and may need adjustment
    if gpu_memory_mb < 4000:  # < 4GB
        base_batch = 1
    elif gpu_memory_mb < 8000:  # 4-8GB
        base_batch = 2
    elif gpu_memory_mb < 12000:  # 8-12GB
        base_batch = 4
    elif gpu_memory_mb < 16000:  # 12-16GB
        base_batch = 8
    elif gpu_memory_mb < 24000:  # 16-24GB
        base_batch = 16
    else:  # 24GB+
        base_batch = 32
    
    logger.info(f"GPU memory: {gpu_memory_mb:.0f}MB, recommended batch size: {base_batch}")
    return base_batch


def scale_learning_rate(base_lr, base_batch_size, new_batch_size, num_gpus=1):
    """Scale learning rate based on batch size using linear scaling rule.
    
    Args:
        base_lr (float): Base learning rate
        base_batch_size (int): Base batch size (from config)
        new_batch_size (int): New batch size per GPU
        num_gpus (int): Number of GPUs
        
    Returns:
        float: Scaled learning rate
    """
    # Linear scaling rule: lr = base_lr * (total_batch_size / base_total_batch_size)
    total_base_batch = base_batch_size
    total_new_batch = new_batch_size * num_gpus
    
    scaled_lr = base_lr * (total_new_batch / total_base_batch)
    
    logger.info(f"Scaling LR: {base_lr} -> {scaled_lr:.6f} "
                f"(batch: {total_base_batch} -> {total_new_batch})")
    
    return scaled_lr


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a detector with automatic configuration tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simplest usage
  python tools/train_auto.py configs/detr/detr_r50_8xb2-150e_coco.py
  
  # With automatic batch size tuning
  python tools/train_auto.py configs/detr/detr_r50_8xb2-150e_coco.py --auto-batch
  
  # Multi-GPU with automatic LR scaling
  python -m torch.distributed.launch --nproc_per_node=4 \\
      tools/train_auto.py configs/detr/detr_r50_8xb2-150e_coco.py --auto-scale-lr
  
  # All features enabled
  python tools/train_auto.py configs/detr/detr_r50_8xb2-150e_coco.py \\
      --auto-batch --auto-scale-lr --amp --wandb-project my_project
        """
    )
    
    # Required arguments
    parser.add_argument(
        'config',
        type=str,
        help='Path to config file'
    )
    
    # Output arguments
    parser.add_argument(
        '--work-dir',
        type=str,
        default=None,
        help='Directory to save logs and models (default: ./work_dirs/<config_name>)'
    )
    
    # Automatic tuning arguments
    parser.add_argument(
        '--auto-batch',
        action='store_true',
        help='Automatically tune batch size based on GPU memory'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size per GPU (default: from config or auto-tuned)'
    )
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='Automatically scale learning rate based on batch size and num GPUs'
    )
    parser.add_argument(
        '--learning-rate',
        '--lr',
        type=float,
        default=None,
        help='Override learning rate (default: from config or auto-scaled)'
    )
    
    # Training arguments
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Enable automatic mixed precision training'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )
    
    # Distributed training
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher (default: auto-detect from environment)'
    )
    parser.add_argument(
        '--local_rank', '--local-rank',
        type=int,
        default=0,
        help='Local rank for distributed training'
    )
    
    # Logging arguments
    parser.add_argument(
        '--wandb-project',
        type=str,
        default=None,
        help='W&B project name (enables wandb logging)'
    )
    parser.add_argument(
        '--wandb-name',
        type=str,
        default=None,
        help='W&B run name'
    )
    
    # Config overrides
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        default=[],
        help='Override config options (e.g., key=value key2=value2)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect launcher from environment
    if args.launcher == 'none':
        if 'SLURM_JOB_ID' in os.environ:
            args.launcher = 'slurm'
        elif 'LOCAL_RANK' in os.environ or 'RANK' in os.environ:
            args.launcher = 'pytorch'
    
    # Set LOCAL_RANK environment variable
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    return args


def main():
    args = parse_args()
    
    # Setup cache for dynamo
    setup_cache_size_limit_of_dynamo()
    
    # Determine number of GPUs
    num_gpus = 1
    if args.launcher != 'none':
        if 'WORLD_SIZE' in os.environ:
            num_gpus = int(os.environ['WORLD_SIZE'])
        logger.info(f"Distributed training with {num_gpus} GPUs")
    
    # Create wrapper with MM config
    logger.info(f"Loading configuration: {args.config}")
    wrapper = HydraWrapper(
        config_path=args.config,
        work_dir=args.work_dir
    )
    
    # Load configuration
    config = wrapper.load_config()
    
    # Set work directory
    if args.work_dir is None:
        config_name = Path(args.config).stem
        args.work_dir = f'./work_dirs/{config_name}'
    config.work_dir = args.work_dir
    logger.info(f"Work directory: {args.work_dir}")
    
    # Get base batch size and learning rate from config
    base_batch_size = getattr(config.train_dataloader, 'batch_size', 2)
    base_lr = getattr(config.optim_wrapper.optimizer, 'lr', 0.0001)
    
    # Automatic batch size tuning
    if args.auto_batch and args.batch_size is None:
        logger.info("Automatic batch size tuning enabled")
        optimal_batch = estimate_optimal_batch_size(config, num_gpus=num_gpus)
        if optimal_batch is not None:
            args.batch_size = optimal_batch
    
    # Apply batch size override
    if args.batch_size is not None:
        logger.info(f"Setting batch size to {args.batch_size} per GPU")
        config.train_dataloader.batch_size = args.batch_size
        
        # Also update val/test batch size if not specified differently
        if hasattr(config, 'val_dataloader'):
            config.val_dataloader.batch_size = args.batch_size
        if hasattr(config, 'test_dataloader'):
            config.test_dataloader.batch_size = args.batch_size
    
    # Automatic learning rate scaling
    if args.auto_scale_lr and args.learning_rate is None:
        logger.info("Automatic learning rate scaling enabled")
        new_batch_size = args.batch_size if args.batch_size else base_batch_size
        args.learning_rate = scale_learning_rate(
            base_lr, base_batch_size, new_batch_size, num_gpus
        )
    
    # Apply learning rate override
    if args.learning_rate is not None:
        logger.info(f"Setting learning rate to {args.learning_rate}")
        config.optim_wrapper.optimizer.lr = args.learning_rate
    
    # Enable automatic mixed precision
    if args.amp:
        logger.info("Enabling automatic mixed precision training")
        config.optim_wrapper.type = 'AmpOptimWrapper'
        config.optim_wrapper.loss_scale = 'dynamic'
    
    # Set launcher
    config.launcher = args.launcher
    
    # Set random seed
    if args.seed is not None:
        config.seed = args.seed
        logger.info(f"Setting random seed to {args.seed}")
    
    # Resume from checkpoint
    if args.resume is not None:
        config.resume = True
        config.load_from = args.resume
        logger.info(f"Resuming from checkpoint: {args.resume}")
    
    # Configure W&B logging
    if args.wandb_project is not None:
        if not WANDB_AVAILABLE:
            logger.warning("W&B requested but not available")
        else:
            logger.info(f"Enabling W&B logging: {args.wandb_project}")
            
            # Initialize wandb config
            wandb_init_kwargs = {
                'project': args.wandb_project,
            }
            if args.wandb_name:
                wandb_init_kwargs['name'] = args.wandb_name
            
            # Update config to use wandb
            if not hasattr(config, 'visualizer'):
                config.visualizer = dict(type='DetLocalVisualizer')
            
            if not hasattr(config.visualizer, 'vis_backends'):
                config.visualizer['vis_backends'] = []
            
            # Add wandb backend
            wandb_backend = dict(
                type='WandbVisBackend',
                init_kwargs=wandb_init_kwargs
            )
            config.visualizer['vis_backends'].append(wandb_backend)
    
    # Apply additional config options
    if args.cfg_options:
        logger.info(f"Applying config overrides: {args.cfg_options}")
        overrides = {}
        for opt in args.cfg_options:
            if '=' in opt:
                key, value = opt.split('=', 1)
                # Try to parse value
                try:
                    value = eval(value)
                except:
                    pass  # Keep as string
                overrides[key] = value
        
        config = wrapper.merge_config(overrides)
    
    # Print final configuration summary
    logger.info("=" * 80)
    logger.info("Training Configuration Summary:")
    logger.info(f"  Config file: {args.config}")
    logger.info(f"  Work directory: {args.work_dir}")
    logger.info(f"  Batch size per GPU: {config.train_dataloader.batch_size}")
    logger.info(f"  Number of GPUs: {num_gpus}")
    logger.info(f"  Total batch size: {config.train_dataloader.batch_size * num_gpus}")
    logger.info(f"  Learning rate: {config.optim_wrapper.optimizer.lr}")
    logger.info(f"  AMP enabled: {args.amp}")
    logger.info(f"  Launcher: {args.launcher}")
    if args.wandb_project:
        logger.info(f"  W&B project: {args.wandb_project}")
    logger.info("=" * 80)
    
    # Start training
    logger.info("Starting training...")
    results = wrapper.train(config)
    
    logger.info("Training completed!")
    logger.info(f"Results: {results}")


if __name__ == '__main__':
    main()
