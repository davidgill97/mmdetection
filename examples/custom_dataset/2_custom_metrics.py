#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
Example 2: Training with Custom Metrics

This example demonstrates how to override and customize evaluation metrics
for your dataset using the MM wrapper system.

Usage:
    python examples/custom_dataset/2_custom_metrics.py \
        --data-root /path/to/your/data \
        --work-dir ./work_dirs/custom_metrics
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
        description='Train with custom metrics configuration'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='/path/to/your/data',
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
        default='./work_dirs/custom_metrics',
        help='Directory to save logs and models'
    )
    parser.add_argument(
        '--metric-type',
        type=str,
        choices=['bbox', 'segm', 'both', 'custom'],
        default='bbox',
        help='Type of metrics to compute'
    )
    return parser.parse_args()


def configure_bbox_metrics(args):
    """Configure bounding box detection metrics."""
    return {
        'val_evaluator': {
            'type': 'CocoMetric',
            'metric': 'bbox',
            'ann_file': f"{args.data_root}/annotations/val.json",
            'classwise': True,  # Report per-class metrics
            'proposal_nums': [100, 300, 1000],  # Proposal counts for AR
            'iou_thrs': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            'metric_items': ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
        }
    }


def configure_segm_metrics(args):
    """Configure instance segmentation metrics."""
    return {
        'val_evaluator': {
            'type': 'CocoMetric',
            'metric': 'segm',
            'ann_file': f"{args.data_root}/annotations/val.json",
            'classwise': True,
            'iou_thrs': [0.5, 0.75, 0.95],
            'metric_items': ['mAP', 'mAP_50', 'mAP_75']
        }
    }


def configure_both_metrics(args):
    """Configure both bbox and segmentation metrics."""
    return {
        'val_evaluator': {
            'type': 'CocoMetric',
            'metric': ['bbox', 'segm'],
            'ann_file': f"{args.data_root}/annotations/val.json",
            'classwise': True
        }
    }


def configure_custom_metrics(args):
    """Configure custom evaluation metrics."""
    return {
        # Multiple evaluators can be configured
        'val_evaluator': [
            # COCO-style metrics
            {
                'type': 'CocoMetric',
                'metric': 'bbox',
                'ann_file': f"{args.data_root}/annotations/val.json",
                'classwise': True,
                'iou_thrs': [0.5, 0.75],  # Only IoU 0.5 and 0.75
                'metric_items': ['mAP', 'mAP_50', 'mAP_75']
            },
            # Can add other metric types here
            # {
            #     'type': 'VOCMetric',
            #     'metric': 'mAP',
            #     'iou_thr': 0.5
            # }
        ]
    }


def main():
    args = parse_args()
    
    logger.info("="*80)
    logger.info("Custom Metrics Configuration Example")
    logger.info("="*80)
    
    # Step 1: Initialize wrapper
    logger.info(f"Loading config: {args.config}")
    wrapper = HydraWrapper(
        config_path=args.config,
        work_dir=args.work_dir
    )
    
    config = wrapper.load_config()
    logger.info("✓ Config loaded")
    
    # Step 2: Configure metrics based on type
    logger.info(f"\nConfiguring metrics: {args.metric_type}")
    
    if args.metric_type == 'bbox':
        metrics_config = configure_bbox_metrics(args)
        logger.info("  Metrics: Bounding box detection (mAP, AR, per-class)")
    elif args.metric_type == 'segm':
        metrics_config = configure_segm_metrics(args)
        logger.info("  Metrics: Instance segmentation (mAP, per-class)")
    elif args.metric_type == 'both':
        metrics_config = configure_both_metrics(args)
        logger.info("  Metrics: Both bbox and segmentation")
    else:  # custom
        metrics_config = configure_custom_metrics(args)
        logger.info("  Metrics: Custom configuration with multiple evaluators")
    
    # Step 3: Apply metrics configuration
    config = wrapper.merge_config(metrics_config)
    logger.info("✓ Metrics configured")
    
    # Step 4: Display metrics summary
    logger.info("\nMetrics Configuration:")
    if isinstance(config.val_evaluator, list):
        logger.info(f"  Number of evaluators: {len(config.val_evaluator)}")
        for i, evaluator in enumerate(config.val_evaluator):
            logger.info(f"  Evaluator {i+1}:")
            logger.info(f"    Type: {evaluator['type']}")
            logger.info(f"    Metric: {evaluator.get('metric', 'N/A')}")
            logger.info(f"    Classwise: {evaluator.get('classwise', False)}")
    else:
        logger.info(f"  Type: {config.val_evaluator['type']}")
        logger.info(f"  Metric: {config.val_evaluator.get('metric', 'N/A')}")
        logger.info(f"  Classwise: {config.val_evaluator.get('classwise', False)}")
        if 'iou_thrs' in config.val_evaluator:
            logger.info(f"  IoU thresholds: {config.val_evaluator['iou_thrs']}")
        if 'metric_items' in config.val_evaluator:
            logger.info(f"  Metric items: {config.val_evaluator['metric_items']}")
    
    # Step 5: Example - Add dataset configuration
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
        }
    }
    config = wrapper.merge_config(dataset_config)
    
    logger.info("\nDataset Configuration:")
    logger.info(f"  Data root: {args.data_root}")
    logger.info(f"  Training annotations: annotations/train.json")
    logger.info(f"  Validation annotations: annotations/val.json")
    
    # Step 6: Demonstrate programmatic metric override
    logger.info("\n" + "="*80)
    logger.info("Programmatic Metric Override Example")
    logger.info("="*80)
    
    # Override with custom IoU thresholds
    custom_override = {
        'val_evaluator': {
            'type': 'CocoMetric',
            'metric': 'bbox',
            'ann_file': f"{args.data_root}/annotations/val.json",
            'classwise': True,
            'iou_thrs': [0.5],  # Only mAP@0.5
            'metric_items': ['mAP_50']  # Only report mAP@0.5
        }
    }
    
    config_override = wrapper.merge_config(custom_override)
    logger.info("✓ Metric overridden to compute only mAP@0.5")
    logger.info("  This is useful for quick validation during development")
    
    # Step 7: Show how to add per-class metric names
    logger.info("\n" + "="*80)
    logger.info("Per-Class Metric Names Example")
    logger.info("="*80)
    
    class_names = ['person', 'vehicle', 'animal', 'object']
    class_config = {
        'train_dataloader': {
            'dataset': {
                'metainfo': {
                    'classes': class_names
                }
            }
        },
        'val_dataloader': {
            'dataset': {
                'metainfo': {
                    'classes': class_names
                }
            }
        }
    }
    
    config_with_classes = wrapper.merge_config(class_config)
    logger.info(f"✓ Class names configured: {class_names}")
    logger.info("  Metrics will be reported per class with these names")
    
    logger.info("\n" + "="*80)
    logger.info("Configuration complete!")
    logger.info("="*80)
    logger.info("\nTo start training with these metrics:")
    logger.info(f"  wrapper.train(config)")
    logger.info("\nOr use Hydra config file for easier management:")
    logger.info(f"  See examples/custom_dataset/configs/custom_metrics.yaml")


if __name__ == '__main__':
    main()
