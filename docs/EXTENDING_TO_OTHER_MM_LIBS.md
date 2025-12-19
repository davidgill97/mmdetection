# Extending MM Wrapper System to Other MM Libraries

This guide shows how to use the wrapper system with other MM libraries like mmsegmentation, mmclassification, etc.

## Overview

The wrapper system is designed to work with any MM library that uses mmengine and mmcv. The core components are library-agnostic and can be used with minimal configuration.

## Supported Libraries

- `mmdet` - Object Detection (already integrated)
- `mmseg` - Semantic Segmentation
- `mmcls` - Image Classification
- `mmpose` - Pose Estimation
- `mmaction` - Action Recognition
- `mmocr` - Optical Character Recognition
- `mmtrack` - Object Tracking

## Using with Other MM Libraries

### Method 1: Direct Config Usage (Simplest)

You can use the wrappers directly with any MM library config:

```bash
# For mmsegmentation
python tools/train_hydra.py \
    --mm-config /path/to/mmsegmentation/configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py \
    --work-dir ./work_dirs/mmseg_training

# For mmclassification
python tools/train_hydra.py \
    --mm-config /path/to/mmclassification/configs/resnet/resnet50_8xb32_in1k.py \
    --work-dir ./work_dirs/mmcls_training
```

### Method 2: Using ConfigManager

For more control and validation:

```python
from mmdet.wrappers import HydraWrapper
from mmdet.wrappers.config_utils import ConfigManager

# Create manager for specific library
manager = ConfigManager(library='mmseg')

# Load and validate config
config = manager.load_config('path/to/mmseg/config.py')
is_valid = manager.validate_config(config)

# Use with wrapper
wrapper = HydraWrapper(
    config_path='path/to/mmseg/config.py',
    work_dir='./work_dirs/mmseg'
)

results = wrapper.train()
```

### Method 3: Creating Library-Specific Hydra Configs

Create dedicated Hydra configs for other MM libraries:

#### Example: MMSegmentation Config

```yaml
# examples/hydra_configs/mmseg_pspnet.yaml
defaults:
  - _self_

# MM config for PSPNet
mm_config: /path/to/mmsegmentation/configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py

# Work directory
work_dir: ./work_dirs/mmseg_pspnet/${now:%Y%m%d_%H%M%S}

# Wandb configuration
wandb:
  project: mmsegmentation
  name: pspnet_cityscapes
  tags:
    - pspnet
    - cityscapes
    - segmentation

# Training settings
train:
  max_iters: 40000
  
# Optimizer settings
optimizer:
  lr: 0.01

# Data settings
data:
  train_batch_size: 2
  val_batch_size: 1
```

#### Example: MMClassification Config

```yaml
# examples/hydra_configs/mmcls_resnet.yaml
defaults:
  - _self_

mm_config: /path/to/mmclassification/configs/resnet/resnet50_8xb32_in1k.py

work_dir: ./work_dirs/mmcls_resnet/${now:%Y%m%d_%H%M%S}

wandb:
  project: mmclassification
  name: resnet50_imagenet
  tags:
    - resnet50
    - imagenet

optimizer:
  lr: 0.1

data:
  train_batch_size: 32
```

## Ray Tune with Other MM Libraries

The Ray wrapper works seamlessly with any MM library:

```bash
# Hyperparameter tuning for mmsegmentation
python tools/train_ray.py \
    --config /path/to/mmsegmentation/configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py \
    --num-samples 20 \
    --metric mIoU \
    --mode max

# Hyperparameter tuning for mmclassification
python tools/train_ray.py \
    --config /path/to/mmclassification/configs/resnet/resnet50_8xb32_in1k.py \
    --num-samples 15 \
    --metric accuracy \
    --mode max
```

## Creating Library-Specific Wrappers (Optional)

For advanced use cases, you can create library-specific wrappers:

```python
# mmdet/wrappers/mmseg_wrapper.py
from .hydra_wrapper import HydraWrapper

class MMSegWrapper(HydraWrapper):
    """Specialized wrapper for mmsegmentation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.library = 'mmseg'
    
    def setup_logging(self, config):
        """Override for segmentation-specific logging."""
        super().setup_logging(config)
        # Add segmentation-specific metrics to log
        if hasattr(config, 'evaluation'):
            config.evaluation['metrics'] = ['mIoU', 'mDice']
```

## Common Patterns

### 1. Multi-Task Training

Train multiple models from different libraries:

```python
from mmdet.wrappers import HydraWrapper

# Detection
det_wrapper = HydraWrapper(config_path='path/to/mmdet/config.py')
det_wrapper.train()

# Segmentation
seg_wrapper = HydraWrapper(config_path='path/to/mmseg/config.py')
seg_wrapper.train()

# Classification
cls_wrapper = HydraWrapper(config_path='path/to/mmcls/config.py')
cls_wrapper.train()
```

### 2. Unified Experiment Tracking

Use consistent wandb configuration across libraries:

```yaml
# shared_wandb_config.yaml
wandb:
  entity: my_team
  project: multi_modal_project
  tags:
    - experiment_v1
```

Include in all library configs:

```yaml
defaults:
  - shared_wandb_config
  - _self_
```

### 3. Shared Hyperparameter Search

Define search spaces that work across libraries:

```yaml
# shared_search_space.yaml

# Learning rate (common to all)
optim_wrapper.optimizer.lr:
  type: loguniform
  min: 0.00001
  max: 0.01

# Batch size (adjust per library)
train_dataloader.batch_size:
  type: choice
  choices: [8, 16, 32]  # Adjust based on memory requirements
```

## Library-Specific Considerations

### MMSegmentation
- Use `mIoU` as primary metric
- Typically uses iteration-based training (not epochs)
- May require larger batch sizes for stable training

### MMClassification
- Use `accuracy` or `top-k accuracy` as metrics
- Usually uses epoch-based training
- Consider learning rate warmup

### MMPose
- Use `PCK` (Percentage of Correct Keypoints) as metric
- May require specific data augmentation

### MMTracking
- Use `MOTA` (Multiple Object Tracking Accuracy) as metric
- Requires video datasets

## Best Practices

1. **Use ConfigManager for validation**: Helps catch configuration errors early
2. **Keep library configs separate**: Organize by library in Hydra config directories
3. **Use consistent wandb projects**: One project per MM library for better organization
4. **Document library-specific settings**: Comment on library-specific parameters
5. **Test with small datasets first**: Verify setup before full training

## Examples

See the `examples/` directory for more examples:
- `examples/hydra_configs/` - Hydra configuration templates
- `examples/wrapper_usage_examples.py` - Programmatic usage examples

## Need Help?

- Main wrapper guide: [WRAPPER_GUIDE.md](./WRAPPER_GUIDE.md)
- Quick start: [QUICKSTART_WRAPPER.md](./QUICKSTART_WRAPPER.md)
- Open an issue on GitHub for library-specific questions
