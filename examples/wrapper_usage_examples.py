#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
Example script demonstrating the usage of MM wrappers.

This script shows how to use both Hydra and Ray wrappers programmatically.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmdet.wrappers import HydraWrapper, RayWrapper
from mmdet.wrappers.config_utils import ConfigConverter, ConfigManager


def example_hydra_wrapper():
    """Example of using HydraWrapper."""
    print("\n" + "="*80)
    print("Example 1: Using HydraWrapper with MM Config")
    print("="*80)
    
    # Initialize wrapper with MM config directly
    wrapper = HydraWrapper(
        config_path='projects/example_project/configs/faster-rcnn_dummy-resnet_fpn_1x_coco.py',
        config_name=None,  # No Hydra config, using MM config directly
        work_dir='./work_dirs/example_hydra'
    )
    
    # Load configuration
    config = wrapper.load_config()
    print(f"✓ Config loaded successfully")
    print(f"  - Model type: {config.model.type}")
    work_dir = config.get('work_dir', wrapper.work_dir)
    print(f"  - Work directory: {work_dir}")
    
    # Apply overrides
    overrides = {
        'optimizer.lr': 0.0001,
        'train_dataloader.batch_size': 4
    }
    config = wrapper.merge_config(overrides)
    print(f"✓ Applied overrides")
    print(f"  - LR: {config.get('optimizer', {}).get('lr', 'N/A')}")
    
    return config


def example_hydra_with_yaml():
    """Example of using HydraWrapper with Hydra YAML config."""
    print("\n" + "="*80)
    print("Example 2: Using HydraWrapper with Hydra Config")
    print("="*80)
    
    try:
        # Initialize wrapper with Hydra config
        wrapper = HydraWrapper(
            config_path='examples/hydra_configs',
            config_name='deformable_detr_mnm',
            work_dir='./work_dirs/example_hydra_yaml'
        )
        
        # Load configuration
        config = wrapper.load_config()
        print(f"✓ Hydra config loaded successfully")
        print(f"  - Model type: {config.model.type if hasattr(config, 'model') else 'N/A'}")
        
        # Get Hydra config
        hydra_cfg = wrapper.get_hydra_config()
        if hydra_cfg:
            print(f"✓ Hydra config object available")
        
        return config
    except Exception as e:
        print(f"✗ Error loading Hydra config: {e}")
        print("  Note: This is expected if Hydra is not installed or config path is incorrect")
        return None


def example_config_converter():
    """Example of using ConfigConverter."""
    print("\n" + "="*80)
    print("Example 3: Using ConfigConverter")
    print("="*80)
    
    from mmengine.config import Config
    
    # Create a simple MM config
    config_dict = {
        'model': {'type': 'DeformableDETR'},
        'optimizer': {'lr': 0.0002},
        'train_dataloader': {'batch_size': 8}
    }
    
    # Convert to MM Config
    mm_config = ConfigConverter.dict_to_mm(config_dict)
    print(f"✓ Converted dict to MM Config")
    print(f"  - Model type: {mm_config.model.type}")
    
    # Convert back to dict
    back_to_dict = ConfigConverter.mm_to_dict(mm_config)
    print(f"✓ Converted MM Config back to dict")
    print(f"  - Keys: {list(back_to_dict.keys())}")
    
    return mm_config


def example_config_manager():
    """Example of using ConfigManager."""
    print("\n" + "="*80)
    print("Example 4: Using ConfigManager for Multi-Library Support")
    print("="*80)
    
    # Create manager for mmdetection
    manager = ConfigManager(library='mmdet')
    print(f"✓ Created ConfigManager for '{manager.library}'")
    print(f"  - Default scope: {manager.get_default_scope()}")
    print(f"  - Supported libraries: {manager.SUPPORTED_LIBRARIES}")
    
    # Validate a config
    config = example_hydra_wrapper()
    is_valid = manager.validate_config(config)
    print(f"✓ Config validation: {'Valid' if is_valid else 'Invalid'}")
    
    return manager


def example_ray_wrapper_setup():
    """Example of setting up RayWrapper (without actually running training)."""
    print("\n" + "="*80)
    print("Example 5: Setting up RayWrapper (setup only, no training)")
    print("="*80)
    
    try:
        from ray import tune
        
        # Define search space
        search_space = {
            'optim_wrapper.optimizer.lr': tune.loguniform(1e-5, 1e-3),
            'train_dataloader.batch_size': tune.choice([4, 8, 16])
        }
        
        # Initialize wrapper
        wrapper = RayWrapper(
            config_path='configs/neurocle/deformable_detr/deformable_detr_r50_1xb2-15e_mnm.py',
            work_dir='./work_dirs/example_ray',
            search_space=search_space,
            num_samples=5,
            metric='coco/bbox_mAP',
            mode='max'
        )
        
        print(f"✓ RayWrapper initialized successfully")
        print(f"  - Search space parameters: {list(search_space.keys())}")
        print(f"  - Number of samples: {wrapper.num_samples}")
        print(f"  - Optimization metric: {wrapper.metric} ({wrapper.mode})")
        
        # Load config (without training)
        config = wrapper.load_config()
        print(f"✓ Base config loaded")
        
        return wrapper
    except ImportError:
        print(f"✗ Ray not installed. Install with: pip install ray[tune]")
        return None
    except Exception as e:
        print(f"✗ Error setting up RayWrapper: {e}")
        return None


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("MM Wrapper System - Usage Examples")
    print("="*80)
    
    # Example 1: Basic Hydra wrapper with MM config
    example_hydra_wrapper()
    
    # Example 2: Hydra wrapper with Hydra YAML config
    example_hydra_with_yaml()
    
    # Example 3: Config converter utilities
    example_config_converter()
    
    # Example 4: Config manager for multi-library support
    example_config_manager()
    
    # Example 5: Ray wrapper setup
    example_ray_wrapper_setup()
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)
    print("\nNext steps:")
    print("1. Try training with Hydra: python tools/train_hydra.py --mm-config <config_path>")
    print("2. Try Ray Tune: python tools/train_ray.py --config <config_path> --single-run")
    print("3. See docs/WRAPPER_GUIDE.md for detailed documentation")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
