#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
Example 1: Basic Custom Dataset Training

This example demonstrates how to train a model on a custom dataset using the
MM wrapper system with minimal configuration.

Usage:
    python examples/custom_dataset/1_basic_custom_dataset.py \
        --data-root /path/to/your/data \
        --ann-file train.json \
        --work-dir ./work_dirs/custom_training
"""

import argparse
import logging
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
        description='Train on custom dataset - Basic Example'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Root directory of your dataset'
    )
    parser.add_argument(
        '--ann-file',
        type=str,
        default='annotations/train.json',
        help='Path to annotation file (relative to data-root)'
    )
    parser.add_argument(
        '--val-ann-file',
        type=str,
        default='annotations/val.json',
        help='Path to validation annotation file'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=80,
        help='Number of classes in your dataset'
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
        default='./work_dirs/custom_dataset',
        help='Directory to save logs and models'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='Learning rate'
    )
    parser.add_argument(
        '--val-interval',
        type=int,
        default=5,
        help='Validation interval (epochs)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("="*80)
    logger.info("Basic Custom Dataset Training Example")
    logger.info("="*80)
    
    # Step 1: Initialize the wrapper with base config
    logger.info(f"Loading base config: {args.config}")
    wrapper = HydraWrapper(
        config_path=args.config,
        work_dir=args.work_dir
    )
    
    # Step 2: Load the configuration
    config = wrapper.load_config()
    logger.info("✓ Base config loaded")
    
    # Step 3: Customize for your dataset
    logger.info(f"Configuring for custom dataset at: {args.data_root}")
    
    custom_config = {
        # Dataset configuration
        'train_dataloader': {
            'batch_size': args.batch_size,
            'dataset': {
                'data_root': args.data_root,
                'ann_file': args.ann_file,
                'data_prefix': {'img': 'images/train/'}
            }
        },
        'val_dataloader': {
            'batch_size': args.batch_size,
            'dataset': {
                'data_root': args.data_root,
                'ann_file': args.val_ann_file,
                'data_prefix': {'img': 'images/val/'}
            }
        },
        'test_dataloader': {
            'batch_size': args.batch_size,
            'dataset': {
                'data_root': args.data_root,
                'ann_file': args.val_ann_file,
                'data_prefix': {'img': 'images/val/'}
            }
        },
        
        # Model configuration
        'model': {
            'bbox_head': {
                'num_classes': args.num_classes
            }
        },
        
        # Training configuration
        'train_cfg': {
            'max_epochs': args.epochs,
            'val_interval': args.val_interval
        },
        
        # Optimizer configuration
        'optim_wrapper': {
            'optimizer': {
                'lr': args.lr
            }
        },
        
        # Evaluation metrics
        'val_evaluator': {
            'type': 'CocoMetric',
            'metric': 'bbox',
            'ann_file': f"{args.data_root}/{args.val_ann_file}"
        },
        'test_evaluator': {
            'type': 'CocoMetric',
            'metric': 'bbox',
            'ann_file': f"{args.data_root}/{args.val_ann_file}"
        }
    }
    
    # Merge custom configuration
    config = wrapper.merge_config(custom_config)
    logger.info("✓ Custom dataset configuration applied")
    
    # Step 4: Display configuration summary
    logger.info("\nConfiguration Summary:")
    logger.info(f"  Dataset root: {args.data_root}")
    logger.info(f"  Training annotations: {args.ann_file}")
    logger.info(f"  Validation annotations: {args.val_ann_file}")
    logger.info(f"  Number of classes: {args.num_classes}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Validation interval: {args.val_interval} epochs")
    logger.info(f"  Work directory: {args.work_dir}")
    
    # Step 5: Setup logging (wandb, tensorboard, etc.)
    wrapper.setup_logging(config)
    logger.info("✓ Logging configured")
    
    # Step 6: Start training
    logger.info("\nStarting training...")
    logger.info("="*80)
    
    try:
        results = wrapper.train(config)
        
        logger.info("="*80)
        logger.info("Training completed successfully!")
        logger.info(f"Results: {results}")
        logger.info(f"Models and logs saved to: {args.work_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
