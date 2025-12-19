#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Training script using Hydra configuration system.

This script provides a Hydra-based training interface for mmdetection that
allows for hierarchical configuration composition and easy parameter overrides.

Example usage:
    # Basic training with Hydra config
    python tools/train_hydra.py --config-path ../examples/hydra_configs --config-name base_config
    
    # With overrides
    python tools/train_hydra.py --config-path ../examples/hydra_configs --config-name base_config \
        wandb.project=my_project work_dir=./my_work_dir
    
    # Using MM config directly
    python tools/train_hydra.py --mm-config configs/neurocle/detr/detr_r50_1xb2-15e_mnm.py
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmdet.wrappers import HydraWrapper
from mmdet.utils import setup_cache_size_limit_of_dynamo

import wandb
wandb.require("core")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a detector with Hydra configuration'
    )
    
    # Hydra-specific arguments
    parser.add_argument(
        '--config-path',
        type=str,
        default=None,
        help='Path to Hydra configuration directory'
    )
    parser.add_argument(
        '--config-name',
        type=str,
        default=None,
        help='Name of Hydra config file (without .yaml extension)'
    )
    parser.add_argument(
        '--overrides',
        nargs='+',
        default=[],
        help='Hydra config overrides (e.g., key=value key2=value2)'
    )
    
    # MM config fallback
    parser.add_argument(
        '--mm-config',
        type=str,
        default=None,
        help='Path to MM config file (used if Hydra config not provided)'
    )
    
    # Common arguments
    parser.add_argument(
        '--work-dir',
        type=str,
        default=None,
        help='Directory to save logs and models'
    )
    parser.add_argument(
        '--version-base',
        type=str,
        default=None,
        help='Hydra version base for compatibility'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='Enable automatic-mixed-precision training'
    )
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


def main():
    args = parse_args()
    
    # Setup cache for dynamo
    setup_cache_size_limit_of_dynamo()
    
    # Determine configuration mode
    if args.config_path and args.config_name:
        # Use Hydra configuration
        config_path = args.config_path
        config_name = args.config_name
        print(f"Using Hydra configuration: {config_path}/{config_name}.yaml")
    elif args.mm_config:
        # Use MM configuration directly
        config_path = args.mm_config
        config_name = None
        print(f"Using MM configuration: {config_path}")
    else:
        raise ValueError(
            "Either --config-path and --config-name (for Hydra) or "
            "--mm-config (for MM config) must be provided"
        )
    
    # Create wrapper
    wrapper = HydraWrapper(
        config_path=config_path,
        config_name=config_name,
        work_dir=args.work_dir,
        overrides=args.overrides,
        version_base=args.version_base,
    )
    
    # Load configuration
    config = wrapper.load_config()
    
    # Apply additional settings
    config.launcher = args.launcher
    
    # Enable automatic mixed precision if requested
    if args.amp:
        config.optim_wrapper.type = 'AmpOptimWrapper'
        config.optim_wrapper.loss_scale = 'dynamic'
    
    # Start training
    print("Starting training...")
    results = wrapper.train(config)
    
    print("Training completed!")
    print(f"Results: {results}")


if __name__ == '__main__':
    main()
