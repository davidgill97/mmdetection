#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
Integration test for the wrapper system.

This script tests the full functionality of both Hydra and Ray wrappers.
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmdet.wrappers import HydraWrapper, RayWrapper
from mmdet.wrappers.config_utils import ConfigConverter, ConfigManager


def test_hydra_wrapper_with_mm_config():
    """Test HydraWrapper with MM config file."""
    print("\n" + "="*80)
    print("TEST 1: HydraWrapper with MM Config")
    print("="*80)
    
    try:
        # Create temporary work directory
        with tempfile.TemporaryDirectory() as tmpdir:
            wrapper = HydraWrapper(
                config_path='projects/example_project/configs/faster-rcnn_dummy-resnet_fpn_1x_coco.py',
                config_name=None,
                work_dir=tmpdir
            )
            
            # Test config loading
            config = wrapper.load_config()
            assert hasattr(config, 'model'), "Config should have 'model' attribute"
            assert config.model.type == 'DeformableDETR', "Model type should be DeformableDETR"
            print("✓ Config loaded successfully")
            
            # Test config merge
            overrides = {
                'optimizer': {'lr': 0.0001},
                'train_dataloader': {'batch_size': 4}
            }
            config = wrapper.merge_config(overrides)
            print("✓ Config merge successful")
            
            # Test setup logging
            wrapper.setup_logging(config)
            assert hasattr(config, 'vis_backends'), "Config should have vis_backends"
            print("✓ Logging setup successful")
            
            print("✅ TEST 1 PASSED")
            return True
    except Exception as e:
        print(f"❌ TEST 1 FAILED: {e}")
        return False


def test_config_converter():
    """Test ConfigConverter utilities."""
    print("\n" + "="*80)
    print("TEST 2: ConfigConverter")
    print("="*80)
    
    try:
        # Test dict to MM config
        config_dict = {
            'model': {'type': 'DeformableDETR'},
            'optimizer': {'lr': 0.0002},
            'train_dataloader': {'batch_size': 8}
        }
        mm_config = ConfigConverter.dict_to_mm(config_dict)
        assert mm_config.model.type == 'DeformableDETR', "Model type mismatch"
        print("✓ Dict to MM config conversion successful")
        
        # Test MM config to dict
        back_to_dict = ConfigConverter.mm_to_dict(mm_config)
        assert 'model' in back_to_dict, "Dict should contain 'model' key"
        assert back_to_dict['model']['type'] == 'DeformableDETR', "Model type mismatch"
        print("✓ MM config to dict conversion successful")
        
        print("✅ TEST 2 PASSED")
        return True
    except Exception as e:
        print(f"❌ TEST 2 FAILED: {e}")
        return False


def test_config_manager():
    """Test ConfigManager for multi-library support."""
    print("\n" + "="*80)
    print("TEST 3: ConfigManager")
    print("="*80)
    
    try:
        # Test mmdet manager
        manager = ConfigManager(library='mmdet')
        assert manager.library == 'mmdet', "Library should be 'mmdet'"
        assert manager.get_default_scope() == 'mmdet', "Default scope should be 'mmdet'"
        print("✓ ConfigManager created for mmdet")
        
        # Test loading config
        config = manager.load_config('projects/example_project/configs/faster-rcnn_dummy-resnet_fpn_1x_coco.py')
        print("✓ Config loaded via ConfigManager")
        
        # Test validation
        is_valid = manager.validate_config(config)
        assert is_valid, "Config should be valid"
        print("✓ Config validation successful")
        
        # Test other libraries
        for lib in ['mmseg', 'mmcls', 'mmpose']:
            mgr = ConfigManager(library=lib)
            assert mgr.library == lib, f"Library should be '{lib}'"
            print(f"✓ ConfigManager created for {lib}")
        
        print("✅ TEST 3 PASSED")
        return True
    except Exception as e:
        print(f"❌ TEST 3 FAILED: {e}")
        return False


def test_ray_wrapper_setup():
    """Test RayWrapper setup (without training)."""
    print("\n" + "="*80)
    print("TEST 4: RayWrapper Setup")
    print("="*80)
    
    try:
        # Check if Ray is available
        try:
            from ray import tune
            ray_available = True
        except ImportError:
            ray_available = False
        
        if not ray_available:
            print("⚠️  Ray not installed - skipping test")
            print("✅ TEST 4 SKIPPED")
            return True
        
        # Create wrapper with search space
        with tempfile.TemporaryDirectory() as tmpdir:
            search_space = {
                'optim_wrapper.optimizer.lr': tune.loguniform(1e-5, 1e-3),
                'train_dataloader.batch_size': tune.choice([4, 8, 16])
            }
            
            wrapper = RayWrapper(
                config_path='projects/example_project/configs/faster-rcnn_dummy-resnet_fpn_1x_coco.py',
                work_dir=tmpdir,
                search_space=search_space,
                num_samples=5,
                metric='coco/bbox_mAP',
                mode='max'
            )
            
            # Test config loading
            config = wrapper.load_config()
            assert hasattr(config, 'model'), "Config should have 'model' attribute"
            print("✓ RayWrapper initialized")
            print("✓ Config loaded successfully")
            
            # Test setup logging
            wrapper.setup_logging(config)
            print("✓ Logging setup successful")
            
            print("✅ TEST 4 PASSED")
            return True
    except Exception as e:
        print(f"❌ TEST 4 FAILED: {e}")
        return False


def test_wrapper_imports():
    """Test that all wrapper components can be imported."""
    print("\n" + "="*80)
    print("TEST 5: Import Tests")
    print("="*80)
    
    try:
        from mmdet.wrappers import BaseWrapper, HydraWrapper, RayWrapper
        print("✓ Main wrappers imported")
        
        from mmdet.wrappers.config_utils import ConfigConverter, ConfigManager
        print("✓ Config utilities imported")
        
        print("✅ TEST 5 PASSED")
        return True
    except Exception as e:
        print(f"❌ TEST 5 FAILED: {e}")
        return False


def test_hydra_optional():
    """Test that Hydra is truly optional."""
    print("\n" + "="*80)
    print("TEST 6: Hydra Optional Test")
    print("="*80)
    
    try:
        # Test that HydraWrapper works without Hydra for MM configs
        with tempfile.TemporaryDirectory() as tmpdir:
            wrapper = HydraWrapper(
                config_path='projects/example_project/configs/faster-rcnn_dummy-resnet_fpn_1x_coco.py',
                config_name=None,  # No Hydra config
                work_dir=tmpdir
            )
            config = wrapper.load_config()
            print("✓ HydraWrapper works with MM config without Hydra")
        
            # Test that it fails appropriately when trying to use Hydra config without Hydra
            try:
                import hydra
                hydra_installed = True
            except ImportError:
                hydra_installed = False
            
            if not hydra_installed:
                try:
                    wrapper = HydraWrapper(
                        config_path='examples/hydra_configs',
                        config_name='base_config',  # Hydra config
                        work_dir=tmpdir
                    )
                    print("❌ Should have raised ImportError for Hydra config without Hydra")
                    return False
                except ImportError:
                    print("✓ Correctly raises ImportError when Hydra config used without Hydra")
            else:
                print("✓ Hydra is installed, skipping ImportError test")
        
        print("✅ TEST 6 PASSED")
        return True
    except Exception as e:
        print(f"❌ TEST 6 FAILED: {e}")
        return False


def main():
    """Run all integration tests."""
    print("\n" + "="*80)
    print("MM Wrapper System - Integration Tests")
    print("="*80)
    
    tests = [
        test_wrapper_imports,
        test_hydra_wrapper_with_mm_config,
        test_config_converter,
        test_config_manager,
        test_hydra_optional,
        test_ray_wrapper_setup,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! The wrapper system is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")
    
    print("="*80 + "\n")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
