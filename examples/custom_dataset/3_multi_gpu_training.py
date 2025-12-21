#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
Example 3: Multi-GPU Distributed Training

This example demonstrates how to train on multiple GPUs using the MM wrapper
system with PyTorch DistributedDataParallel.

Usage:
    # Single GPU
    python examples/custom_dataset/3_multi_gpu_training.py \
        --data-root /path/to/your/data
    
    # Multi-GPU (4 GPUs)
    python -m torch.distributed.launch --nproc_per_node=4 \
        examples/custom_dataset/3_multi_gpu_training.py \
        --data-root /path/to/your/data \
        --launcher pytorch
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mmdet.wrappers import HydraWrapper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Multi-GPU distributed training example'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Root directory of your dataset'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/detr/detr_r50_8xb2-150e_coco.py',
        help='Base MM config file'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        default='./work_dirs/multi_gpu',
        help='Directory to save logs and models'
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher for distributed training'
    )
    parser.add_argument(
        '--local_rank', '--local-rank',
        type=int,
        default=0,
        help='Local rank for distributed training'
    )
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=4,
        help='Number of GPUs for training'
    )
    parser.add_argument(
        '--batch-size-per-gpu',
        type=int,
        default=4,
        help='Batch size per GPU'
    )
    parser.add_argument(
        '--base-lr',
        type=float,
        default=0.0001,
        help='Base learning rate (will be scaled for multi-GPU)'
    )
    return parser.parse_args()


def calculate_lr_for_multi_gpu(base_lr, batch_size_per_gpu, num_gpus, base_batch_size=2):
    """Calculate learning rate for multi-GPU training.
    
    Args:
        base_lr (float): Base learning rate
        batch_size_per_gpu (int): Batch size per GPU
        num_gpus (int): Number of GPUs
        base_batch_size (int): Base batch size used for base_lr
    
    Returns:
        float: Scaled learning rate
    """
    total_batch_size = batch_size_per_gpu * num_gpus
    lr_scale = total_batch_size / base_batch_size
    return base_lr * lr_scale


def main():
    args = parse_args()
    
    # Set LOCAL_RANK environment variable
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    logger.info("="*80)
    logger.info("Multi-GPU Distributed Training Example")
    logger.info("="*80)
    
    # Display distributed training configuration
    if args.launcher != 'none':
        logger.info(f"\nDistributed Training Configuration:")
        logger.info(f"  Launcher: {args.launcher}")
        logger.info(f"  Number of GPUs: {args.num_gpus}")
        logger.info(f"  Batch size per GPU: {args.batch_size_per_gpu}")
        logger.info(f"  Total batch size: {args.batch_size_per_gpu * args.num_gpus}")
        logger.info(f"  Local rank: {args.local_rank}")
    else:
        logger.info("\nSingle GPU Training")
        logger.info(f"  Batch size: {args.batch_size_per_gpu}")
    
    # Calculate scaled learning rate for multi-GPU
    if args.launcher != 'none':
        scaled_lr = calculate_lr_for_multi_gpu(
            args.base_lr,
            args.batch_size_per_gpu,
            args.num_gpus
        )
        logger.info(f"\nLearning Rate Scaling:")
        logger.info(f"  Base LR: {args.base_lr}")
        logger.info(f"  Scaled LR: {scaled_lr}")
        logger.info(f"  Scale factor: {scaled_lr / args.base_lr:.2f}x")
    else:
        scaled_lr = args.base_lr
    
    # Initialize wrapper
    logger.info(f"\nLoading config: {args.config}")
    wrapper = HydraWrapper(
        config_path=args.config,
        work_dir=args.work_dir
    )
    
    config = wrapper.load_config()
    logger.info("✓ Config loaded")
    
    # Configure for distributed training
    multi_gpu_config = {
        # Launcher configuration
        'launcher': args.launcher,
        
        # Dataset configuration
        'train_dataloader': {
            'batch_size': args.batch_size_per_gpu,
            'num_workers': 4,
            'persistent_workers': True,
            'dataset': {
                'data_root': args.data_root,
                'ann_file': 'annotations/train.json'
            }
        },
        'val_dataloader': {
            'batch_size': args.batch_size_per_gpu,
            'num_workers': 4,
            'persistent_workers': True,
            'dataset': {
                'data_root': args.data_root,
                'ann_file': 'annotations/val.json'
            }
        },
        
        # Optimizer with scaled learning rate
        'optim_wrapper': {
            'optimizer': {
                'lr': scaled_lr
            }
        },
        
        # Enable mixed precision for faster training
        'optim_wrapper': {
            'type': 'AmpOptimWrapper' if args.launcher != 'none' else 'OptimWrapper',
            'optimizer': {
                'lr': scaled_lr
            },
            'loss_scale': 'dynamic' if args.launcher != 'none' else None
        },
        
        # Evaluation metrics
        'val_evaluator': {
            'type': 'CocoMetric',
            'metric': 'bbox',
            'ann_file': f"{args.data_root}/annotations/val.json"
        }
    }
    
    # Apply configuration
    config = wrapper.merge_config(multi_gpu_config)
    config.launcher = args.launcher
    logger.info("✓ Multi-GPU configuration applied")
    
    # Display training configuration
    logger.info("\nTraining Configuration:")
    logger.info(f"  Dataset: {args.data_root}")
    logger.info(f"  Batch size per GPU: {args.batch_size_per_gpu}")
    logger.info(f"  Learning rate: {scaled_lr}")
    logger.info(f"  Mixed precision: {args.launcher != 'none'}")
    logger.info(f"  Work directory: {args.work_dir}")
    
    # Setup logging
    wrapper.setup_logging(config)
    logger.info("✓ Logging configured")
    
    # Start training
    logger.info("\n" + "="*80)
    logger.info("Starting training...")
    logger.info("="*80)
    
    if args.launcher != 'none':
        logger.info("\nNote: Make sure you launched this script with:")
        logger.info(f"  python -m torch.distributed.launch --nproc_per_node={args.num_gpus} \\")
        logger.info(f"    {sys.argv[0]} --launcher pytorch ...")
        logger.info("")
    
    try:
        results = wrapper.train(config)
        
        # Only print on rank 0
        if args.local_rank == 0:
            logger.info("="*80)
            logger.info("Training completed successfully!")
            logger.info(f"Results: {results}")
            logger.info(f"Models and logs saved to: {args.work_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
