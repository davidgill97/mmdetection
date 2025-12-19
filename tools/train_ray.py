#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Training script using Ray Tune for hyperparameter optimization.

This script provides a Ray Tune-based training interface for mmdetection that
enables scalable hyperparameter tuning with various search algorithms.

Example usage:
    # Basic hyperparameter tuning
    python tools/train_ray.py --config configs/neurocle/detr/detr_r50_1xb2-15e_mnm.py \
        --num-samples 20 --metric coco/bbox_mAP
    
    # With custom search space
    python tools/train_ray.py --config configs/neurocle/detr/detr_r50_1xb2-15e_mnm.py \
        --search-space search_space.yaml --num-samples 50
    
    # Single training without tuning
    python tools/train_ray.py --config configs/neurocle/detr/detr_r50_1xb2-15e_mnm.py \
        --single-run
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmdet.wrappers import RayWrapper
from mmdet.utils import setup_cache_size_limit_of_dynamo

try:
    from ray import tune
except ImportError:
    tune = None

import wandb
try:
    wandb.require("core")
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available or failed to initialize")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a detector with Ray Tune hyperparameter optimization'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to MM config file'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        default='./ray_results',
        help='Directory to save logs and models'
    )
    
    # Ray Tune arguments
    parser.add_argument(
        '--search-space',
        type=str,
        default=None,
        help='Path to YAML file defining search space'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of trials to run'
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        choices=['asha', 'pbt', 'none'],
        default='asha',
        help='Scheduler type for early stopping'
    )
    parser.add_argument(
        '--search-alg',
        type=str,
        choices=['hyperopt', 'random', 'none'],
        default='hyperopt',
        help='Search algorithm'
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
        help='Optimization mode (minimize or maximize metric)'
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
        default=2,
        help='Number of CPUs per trial'
    )
    
    # Single run mode
    parser.add_argument(
        '--single-run',
        action='store_true',
        help='Run single training without hyperparameter tuning'
    )
    
    # Additional arguments
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher'
    )
    parser.add_argument(
        '--local_rank', '--local-rank',
        type=int,
        default=0,
        help='Local rank for distributed training'
    )
    
    args = parser.parse_args()
    
    # Set LOCAL_RANK environment variable
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    return args


def load_search_space(search_space_path: str) -> dict:
    """Load search space from YAML file.
    
    Args:
        search_space_path (str): Path to YAML file.
        
    Returns:
        dict: Search space configuration.
    """
    with open(search_space_path, 'r') as f:
        search_space = yaml.safe_load(f)
    
    # Convert tune specifications
    if tune is not None:
        converted_space = {}
        for key, value in search_space.items():
            if isinstance(value, dict) and 'type' in value:
                tune_type = value['type']
                if tune_type == 'uniform':
                    converted_space[key] = tune.uniform(value['min'], value['max'])
                elif tune_type == 'loguniform':
                    converted_space[key] = tune.loguniform(value['min'], value['max'])
                elif tune_type == 'choice':
                    converted_space[key] = tune.choice(value['choices'])
                elif tune_type == 'randint':
                    converted_space[key] = tune.randint(value['min'], value['max'])
                else:
                    converted_space[key] = value
            else:
                converted_space[key] = value
        return converted_space
    
    return search_space


def create_default_search_space() -> dict:
    """Create default search space for common hyperparameters.
    
    Returns:
        dict: Default search space.
    """
    if tune is None:
        return {}
    
    return {
        'optim_wrapper.optimizer.lr': tune.loguniform(1e-5, 1e-3),
        'optim_wrapper.optimizer.weight_decay': tune.loguniform(1e-6, 1e-3),
        'train_dataloader.batch_size': tune.choice([4, 8, 16, 32]),
    }


def main():
    args = parse_args()
    
    # Setup cache for dynamo
    setup_cache_size_limit_of_dynamo()
    
    # Load search space
    if args.search_space:
        search_space = load_search_space(args.search_space)
    else:
        search_space = create_default_search_space() if not args.single_run else {}
    
    # Create resources configuration
    resources = {
        'cpu': args.cpus_per_trial,
        'gpu': args.gpus_per_trial,
    }
    
    # Handle scheduler and search algorithm
    scheduler = args.scheduler if args.scheduler != 'none' else None
    search_alg = args.search_alg if args.search_alg != 'none' else None
    
    # Create wrapper
    wrapper = RayWrapper(
        config_path=args.config,
        work_dir=args.work_dir,
        search_space=search_space,
        num_samples=args.num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
        metric=args.metric,
        mode=args.mode,
        resources_per_trial=resources,
    )
    
    # Execute training
    if args.single_run:
        print("Starting single training run...")
        config = wrapper.load_config()
        config.launcher = args.launcher
        results = wrapper.train_single(config)
        print("Training completed!")
        print(f"Results: {results}")
    else:
        print(f"Starting Ray Tune hyperparameter optimization with {args.num_samples} trials...")
        print(f"Search space: {search_space}")
        results = wrapper.train()
        print("Hyperparameter tuning completed!")
        print(f"Best configuration: {results['best_config']}")
        print(f"Best metrics: {results['best_metrics']}")


if __name__ == '__main__':
    main()
