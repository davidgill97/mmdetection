# Extending MM Wrapper System to Other MM Libraries

This guide shows how to use the wrapper system with other MM libraries like mmsegmentation, mmpretrain (classification), mmpose, etc.

## Overview

The wrapper system is designed to work with any MM library that uses mmengine and mmcv. The core components are library-agnostic and can be used with minimal configuration.

## Supported Libraries

- `mmdet` - Object Detection (already integrated)
- `mmseg` - Semantic Segmentation
- `mmcls` - Image Classification (legacy name)
- `mmpretrain` - Image Classification (new name, replaces mmcls)
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

# For mmpretrain (classification)
python tools/train_hydra.py \
    --mm-config /path/to/mmpretrain/configs/resnet/resnet50_8xb32_in1k.py \
    --work-dir ./work_dirs/mmpretrain_training
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

#### Example: MMPretrain (Classification) Config

```yaml
# examples/hydra_configs/mmpretrain_resnet50.yaml
defaults:
  - _self_

mm_config: /path/to/mmpretrain/configs/resnet/resnet50_8xb32_in1k.py

work_dir: ./work_dirs/mmpretrain_resnet/${now:%Y%m%d_%H%M%S}

wandb:
  project: mmpretrain
  name: resnet50_imagenet
  tags:
    - resnet50
    - imagenet
    - classification

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

# Hyperparameter tuning for mmpretrain (classification)
python tools/train_ray.py \
    --config /path/to/mmpretrain/configs/resnet/resnet50_8xb32_in1k.py \
    --num-samples 15 \
    --metric accuracy/top1 \
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

# Classification (mmpretrain)
pretrain_wrapper = HydraWrapper(config_path='path/to/mmpretrain/config.py')
pretrain_wrapper.train()
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
- Use `mIoU` (mean Intersection over Union) as primary metric
- Typically uses iteration-based training (not epochs)
- May require larger batch sizes for stable training
- Common models: PSPNet, DeepLabV3, SegFormer

### MMPretrain (Classification)
- Replaces mmclassification (use `mmpretrain` for new projects)
- Use `accuracy/top1` or `accuracy/top5` as metrics
- Usually uses epoch-based training
- Consider learning rate warmup for large batch sizes
- Common models: ResNet, Vision Transformer, Swin Transformer

### MMPose
- Use `PCK` (Percentage of Correct Keypoints) as metric
- May require specific data augmentation
- Common models: HRNet, SimCC

### MMTracking
- Use `MOTA` (Multiple Object Tracking Accuracy) as metric
- Requires video datasets
- Common models: ByteTrack, SORT

## Complete Examples

### Example 1: MMSegmentation with Multi-GPU

```python
from mmdet.wrappers import HydraWrapper

# Create wrapper for PSPNet segmentation
wrapper = HydraWrapper(
    config_path='/path/to/mmsegmentation/configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py',
    work_dir='./work_dirs/mmseg_training'
)

# Load config
config = wrapper.load_config()

# Configure for multi-GPU training
config.launcher = 'pytorch'
config.train_dataloader.batch_size = 2  # Per GPU

# Launch with: python -m torch.distributed.launch --nproc_per_node=4
results = wrapper.train(config)
```

### Example 2: MMPretrain for Image Classification

```python
from mmdet.wrappers import HydraWrapper

# Create wrapper for ResNet50 classification
wrapper = HydraWrapper(
    config_path='/path/to/mmpretrain/configs/resnet/resnet50_8xb32_in1k.py',
    work_dir='./work_dirs/mmpretrain_training'
)

# Load and configure
config = wrapper.load_config()

# Override for custom dataset
config = wrapper.merge_config({
    'data_root': '/path/to/your/dataset',
    'train_dataloader': {
        'dataset': {
            'data_prefix': 'train',
            'ann_file': 'train.txt'
        }
    },
    'val_dataloader': {
        'dataset': {
            'data_prefix': 'val',
            'ann_file': 'val.txt'
        }
    }
})

results = wrapper.train(config)
```

### Example 3: Combined Detection and Segmentation

```python
from mmdet.wrappers import HydraWrapper, ConfigManager

# Train detection model
det_manager = ConfigManager(library='mmdet')
det_wrapper = HydraWrapper(
    config_path='path/to/mmdet/detr_config.py',
    work_dir='./work_dirs/detection'
)
det_results = det_wrapper.train()

# Train segmentation model
seg_manager = ConfigManager(library='mmseg')
seg_wrapper = HydraWrapper(
    config_path='path/to/mmseg/pspnet_config.py',
    work_dir='./work_dirs/segmentation'
)
seg_results = seg_wrapper.train()

print(f"Detection mAP: {det_results.get('mAP', 'N/A')}")
print(f"Segmentation mIoU: {seg_results.get('mIoU', 'N/A')}")
```

## Best Practices

1. **Use ConfigManager for validation**: Helps catch configuration errors early
2. **Keep library configs separate**: Organize by library in Hydra config directories
3. **Use consistent wandb projects**: One project per MM library for better organization
4. **Document library-specific settings**: Comment on library-specific parameters
5. **Test with small datasets first**: Verify setup before full training
6. **Use mmpretrain instead of mmcls**: For new classification projects

## Examples

See the `examples/` directory for more examples:
- `examples/hydra_configs/` - Hydra configuration templates
  - `mmseg_pspnet.yaml` - MMSegmentation example
  - `mmpretrain_resnet50.yaml` - MMPretrain example
- `examples/wrapper_usage_examples.py` - Programmatic usage examples

## Need Help?

- Main wrapper guide: [WRAPPER_GUIDE.md](./WRAPPER_GUIDE.md)
- Quick start: [QUICKSTART_WRAPPER.md](./QUICKSTART_WRAPPER.md)
- Open an issue on GitHub for library-specific questions
