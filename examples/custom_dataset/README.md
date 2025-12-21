# Custom Dataset Training Examples

This directory contains complete examples for training on custom datasets using the MM wrapper system.

## Overview

These examples demonstrate how to:
- Prepare and register custom datasets
- Configure training with custom metrics
- Use Hydra for flexible configuration management
- Run hyperparameter tuning with Ray Tune
- Train on multiple GPUs
- Export models for deployment

## Directory Structure

```
custom_dataset/
├── README.md                           # This file
├── 1_basic_custom_dataset.py          # Basic custom dataset training
├── 2_custom_metrics.py                # Training with custom metrics
├── 3_multi_gpu_training.py            # Multi-GPU distributed training
├── 4_hyperparameter_tuning.py         # Ray Tune hyperparameter optimization
├── 5_complete_pipeline.py             # Complete training-to-deployment pipeline
├── configs/
│   ├── custom_dataset_base.yaml       # Base Hydra config for custom dataset
│   ├── custom_metrics.yaml            # Config with custom metrics
│   └── multi_gpu.yaml                 # Multi-GPU configuration
└── datasets/
    └── custom_dataset.py              # Custom dataset class example
```

## Quick Start

### 1. Basic Custom Dataset Training

Train a model on your custom dataset with minimal configuration:

```bash
python examples/custom_dataset/1_basic_custom_dataset.py \
    --data-root /path/to/your/data \
    --ann-file train.json \
    --work-dir ./work_dirs/custom_training
```

### 2. Custom Metrics

Override evaluation metrics in your config:

```bash
python examples/custom_dataset/2_custom_metrics.py \
    --config examples/custom_dataset/configs/custom_metrics.yaml
```

Or programmatically:

```python
from mmdet.wrappers import HydraWrapper

wrapper = HydraWrapper(config_path='configs/detr/detr_r50_8xb2-150e_coco.py')
config = wrapper.load_config()

# Override metrics
config = wrapper.merge_config({
    'val_evaluator': {
        'type': 'CocoMetric',
        'metric': ['bbox', 'segm'],
        'classwise': True
    }
})

results = wrapper.train(config)
```

### 3. Multi-GPU Training

Train on 4 GPUs with automatic batch size scaling:

```bash
python -m torch.distributed.launch --nproc_per_node=4 \
    examples/custom_dataset/3_multi_gpu_training.py \
    --launcher pytorch
```

### 4. Hyperparameter Tuning

Find optimal hyperparameters with Ray Tune:

```bash
python examples/custom_dataset/4_hyperparameter_tuning.py \
    --num-samples 20 \
    --gpus-per-trial 1
```

### 5. Complete Pipeline

Full pipeline from training to deployment:

```bash
python examples/custom_dataset/5_complete_pipeline.py \
    --data-root /path/to/data \
    --export-onnx \
    --export-tensorrt
```

## Dataset Preparation

### COCO Format

Organize your data in COCO format:

```
/path/to/your/data/
├── annotations/
│   ├── train.json
│   └── val.json
└── images/
    ├── train/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── val/
        ├── img3.jpg
        └── img4.jpg
```

Annotation format:
```json
{
  "images": [
    {"id": 1, "file_name": "img1.jpg", "height": 480, "width": 640}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 1000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "class1"},
    {"id": 2, "name": "class2"}
  ]
}
```

### Custom Dataset Class

See `datasets/custom_dataset.py` for a complete example of creating a custom dataset class.

## Configuration

### Base Configuration

```yaml
# examples/custom_dataset/configs/custom_dataset_base.yaml
defaults:
  - base_config
  - _self_

# Dataset configuration
data_root: /path/to/your/data
ann_file_train: annotations/train.json
ann_file_val: annotations/val.json

# Number of classes in your dataset
num_classes: 10

# Training settings
train:
  max_epochs: 100
  val_interval: 10

# Metrics configuration
val_evaluator:
  type: CocoMetric
  metric: bbox
  classwise: true

# Work directory
work_dir: ./work_dirs/custom_dataset/${now:%Y%m%d_%H%M%S}
```

### Custom Metrics Configuration

```yaml
# examples/custom_dataset/configs/custom_metrics.yaml
defaults:
  - custom_dataset_base
  - _self_

# Override metrics
val_evaluator:
  type: CocoMetric
  metric: [bbox, segm]  # Multiple metrics
  classwise: true       # Per-class metrics
  
# Additional metric configurations
test_evaluator:
  type: CocoMetric
  metric: bbox
  format_only: false
  outfile_prefix: ./results/custom_dataset
```

### Multi-GPU Configuration

```yaml
# examples/custom_dataset/configs/multi_gpu.yaml
defaults:
  - custom_dataset_base
  - _self_

# Launcher for distributed training
launcher: pytorch

# Batch size per GPU
data:
  train_batch_size: 4  # With 4 GPUs = 16 total
  val_batch_size: 4
  num_workers: 4

# Learning rate scaling for multi-GPU
optimizer:
  lr: 0.0004  # 0.0001 * 4 GPUs

# Enable mixed precision
amp: true
```

## Advanced Examples

### Custom Evaluation Metrics

Define custom metrics by overriding the evaluator:

```python
from mmdet.wrappers import HydraWrapper

wrapper = HydraWrapper(config_path='your_config.py')
config = wrapper.load_config()

# Add custom metric
config.val_evaluator = {
    'type': 'CocoMetric',
    'metric': ['bbox'],
    'classwise': True,
    'proposal_nums': [100, 300, 1000],  # Custom proposal numbers
    'iou_thrs': [0.5, 0.75, 0.95]       # Custom IoU thresholds
}

results = wrapper.train(config)
```

### Multi-Metric Evaluation

Evaluate with multiple metrics simultaneously:

```python
config.val_evaluator = [
    {'type': 'CocoMetric', 'metric': 'bbox'},
    {'type': 'VOCMetric', 'metric': 'mAP'},
    {'type': 'CityscapesMetric', 'metric': 'mIoU'}
]
```

### Custom Class Names

Map class IDs to custom names:

```python
config.val_evaluator = {
    'type': 'CocoMetric',
    'metric': 'bbox',
    'classwise': True,
}

# Update dataset with custom classes
config.train_dataloader.dataset.metainfo = {
    'classes': ('person', 'vehicle', 'animal', 'object')
}
```

## Tips and Best Practices

### 1. Data Preparation
- Ensure annotations are in correct format
- Verify image paths are correct
- Check class IDs start from 0 or 1 (depending on format)
- Validate annotation files before training

### 2. Metrics Configuration
- Choose appropriate metrics for your task
- Use `classwise=True` for per-class analysis
- Configure IoU thresholds based on your use case
- Test metrics on small validation set first

### 3. Multi-GPU Training
- Scale learning rate proportionally with number of GPUs
- Adjust batch size to maximize GPU utilization
- Use synchronized batch normalization for small batches
- Monitor GPU memory usage

### 4. Hyperparameter Tuning
- Start with small search space
- Use ASHA scheduler for efficient early stopping
- Allocate appropriate GPU resources per trial
- Log all experiments with wandb

### 5. Model Export
- Test export on small example first
- Verify exported model produces same results
- Use appropriate precision (FP16/FP32) for your hardware
- Profile inference speed before deployment

## Troubleshooting

### Dataset Loading Issues

**Problem**: Dataset not found
```
Solution: Check data_root and ann_file paths are correct
```

**Problem**: Annotation format errors
```
Solution: Validate JSON format and required fields
```

### Metric Errors

**Problem**: Metric not computed
```
Solution: Ensure metric type matches your task and data format
```

**Problem**: NaN metrics during training
```
Solution: Check if validation set has valid annotations
```

### Multi-GPU Issues

**Problem**: Out of memory
```
Solution: Reduce batch_size per GPU or use gradient accumulation
```

**Problem**: Different results across GPUs
```
Solution: Ensure synchronized batch normalization is enabled
```

## Additional Resources

- [MM Wrapper System Documentation](../../docs/WRAPPER_GUIDE.md)
- [Multi-GPU Training Guide](../../docs/MULTI_GPU_AND_EXPORT.md)
- [MMDetection Custom Dataset Tutorial](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html)
- [COCO Dataset Format](https://cocodataset.org/#format-data)

## Next Steps

1. Prepare your dataset in COCO format
2. Start with `1_basic_custom_dataset.py` example
3. Customize metrics with `2_custom_metrics.py`
4. Scale to multi-GPU with `3_multi_gpu_training.py`
5. Optimize with `4_hyperparameter_tuning.py`
6. Deploy with `5_complete_pipeline.py`

Happy training! 🚀
