#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
Example 4: Hyperparameter Tuning with Ray Tune

This example demonstrates how to use Ray Tune for automatic hyperparameter
optimization on custom datasets.

Usage:
    python examples/custom_dataset/4_hyperparameter_tuning.py \
        --data-root /path/to/your/data \
        --num-samples 20 \
        --gpus-per-trial 1
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mmdet.wrappers import RayWrapper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning with Ray Tune'
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
        default='./ray_results',
        help='Directory to save Ray Tune results'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=20,
        help='Number of trials to run'
    )
    parser.add_argument(
        '--gpus-per-trial',
        type=float,
        default=1.0,
        help='Number of GPUs per trial'
    )
    parser.add_argument(
        '--cpus-per-trial',
        type=int,
        default=4,
        help='Number of CPUs per trial'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='coco/bbox_mAP',
        help='Metric to optimize'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['min', 'max'],
        default='max',
        help='Optimization mode'
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        choices=['asha', 'pbt', 'none'],
        default='asha',
        help='Scheduler for early stopping'
    )
    return parser.parse_args()


def create_search_space():
    """Define the hyperparameter search space."""
    try:
        from ray import tune
    except ImportError:
        logger.error("Ray not installed. Please install with: pip install ray[tune]")
        sys.exit(1)
    
    # Define search space for hyperparameters
    search_space = {
        # Learning rate - log uniform between 1e-5 and 1e-3
        'optim_wrapper.optimizer.lr': tune.loguniform(1e-5, 1e-3),
        
        # Weight decay - log uniform
        'optim_wrapper.optimizer.weight_decay': tune.loguniform(1e-6, 1e-3),
        
        # Batch size - discrete choices
        'train_dataloader.batch_size': tune.choice([2, 4, 8]),
        
        # Gradient clipping - uniform
        # 'optim_wrapper.clip_grad.max_norm': tune.uniform(0.1, 10.0),
    }
    
    return search_space


def main():
    args = parse_args()
    
    logger.info("="*80)
    logger.info("Hyperparameter Tuning with Ray Tune")
    logger.info("="*80)
    
    # Create search space
    logger.info("\nCreating search space...")
    search_space = create_search_space()
    
    logger.info("Search space parameters:")
    for param, config in search_space.items():
        logger.info(f"  {param}: {config}")
    
    # Initialize Ray wrapper
    logger.info(f"\nInitializing Ray Tune wrapper...")
    logger.info(f"  Base config: {args.config}")
    logger.info(f"  Number of trials: {args.num_samples}")
    logger.info(f"  GPUs per trial: {args.gpus_per_trial}")
    logger.info(f"  CPUs per trial: {args.cpus_per_trial}")
    logger.info(f"  Optimization metric: {args.metric} ({args.mode})")
    logger.info(f"  Scheduler: {args.scheduler}")
    
    wrapper = RayWrapper(
        config_path=args.config,
        work_dir=args.work_dir,
        search_space=search_space,
        num_samples=args.num_samples,
        scheduler=args.scheduler,
        search_alg='hyperopt',
        metric=args.metric,
        mode=args.mode,
        resources_per_trial={
            'cpu': args.cpus_per_trial,
            'gpu': args.gpus_per_trial
        }
    )
    
    # Load and configure base config
    logger.info("\nConfiguring base dataset...")
    config = wrapper.load_config()
    
    # Add dataset configuration
    dataset_config = {
        'train_dataloader': {
            'dataset': {
                'data_root': args.data_root,
                'ann_file': 'annotations/train.json'
            }
        },
        'val_dataloader': {
            'dataset': {
                'data_root': args.data_root,
                'ann_file': 'annotations/val.json'
            }
        },
        'val_evaluator': {
            'type': 'CocoMetric',
            'metric': 'bbox',
            'ann_file': f"{args.data_root}/annotations/val.json"
        }
    }
    
    config = wrapper.merge_config(dataset_config)
    logger.info("✓ Dataset configured")
    
    # Configure custom metrics if needed
    logger.info("\nConfiguring evaluation metrics...")
    config = wrapper.configure_metrics(
        config,
        metric_type='bbox',
        classwise=True,
        iou_thrs=[0.5, 0.75],  # Focus on key IoU thresholds for faster evaluation
        metric_items=['mAP', 'mAP_50', 'mAP_75']
    )
    logger.info("✓ Metrics configured")
    
    # Display tuning configuration
    logger.info("\n" + "="*80)
    logger.info("Starting hyperparameter tuning...")
    logger.info("="*80)
    logger.info(f"\nDataset: {args.data_root}")
    logger.info(f"Results will be saved to: {args.work_dir}")
    logger.info("\nThis may take a while depending on:")
    logger.info(f"  - Number of trials: {args.num_samples}")
    logger.info(f"  - Training time per trial")
    logger.info(f"  - Available GPU resources: {args.gpus_per_trial} per trial")
    
    # Start tuning
    try:
        results = wrapper.train()
        
        logger.info("\n" + "="*80)
        logger.info("Hyperparameter tuning completed!")
        logger.info("="*80)
        
        logger.info("\nBest configuration found:")
        for param, value in results['best_config'].items():
            logger.info(f"  {param}: {value}")
        
        logger.info(f"\nBest metrics:")
        for metric, value in results['best_metrics'].items():
            logger.info(f"  {metric}: {value}")
        
        logger.info(f"\nBest checkpoint: {results.get('best_checkpoint', 'N/A')}")
        logger.info(f"\nAll results saved to: {args.work_dir}")
        
        # Save best config for future use
        import json
        best_config_file = f"{args.work_dir}/best_config.json"
        with open(best_config_file, 'w') as f:
            json.dump(results['best_config'], f, indent=2)
        logger.info(f"\nBest config saved to: {best_config_file}")
        
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        raise


if __name__ == '__main__':
    main()
