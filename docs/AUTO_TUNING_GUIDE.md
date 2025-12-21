# Auto-Tuning Guide: Automatic Batch Size and Learning Rate Optimization

This guide explains how to use the automatic configuration tuning features to optimize training performance and GPU utilization.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Automatic Batch Size Tuning](#automatic-batch-size-tuning)
3. [Automatic Learning Rate Scaling](#automatic-learning-rate-scaling)
4. [Simplified CLI Usage](#simplified-cli-usage)
5. [Programmatic API](#programmatic-api)
6. [Best Practices](#best-practices)
7. [Advanced Usage](#advanced-usage)

## Quick Start

### Simplest Usage (One Command)

```bash
# Just provide the config - everything else is automatic!
python tools/train_auto.py configs/detr/detr_r50_8xb2-150e_coco.py --auto-batch --auto-scale-lr
```

This single command will:
- ✅ Detect available GPU memory
- ✅ Automatically tune batch size to maximize GPU utilization
- ✅ Scale learning rate proportionally to maintain training dynamics
- ✅ Set up proper work directory
- ✅ Configure logging

### Multi-GPU Training Made Easy

```bash
# 4 GPUs with automatic tuning
python -m torch.distributed.launch --nproc_per_node=4 \
    tools/train_auto.py configs/detr/detr_r50_8xb2-150e_coco.py \
    --auto-batch --auto-scale-lr --amp
```

## Automatic Batch Size Tuning

### Overview

The `--auto-batch` flag automatically determines the optimal batch size based on:
- Available GPU memory
- Model architecture complexity
- Input image size
- Number of GPUs

### How It Works

```python
# The system:
# 1. Detects GPU memory (e.g., 24GB for RTX 3090)
# 2. Estimates memory per sample based on model
# 3. Calculates optimal batch size
# 4. Applies conservative margin to prevent OOM

GPU Memory     Typical Batch Size (Medium Model)
-----------    ----------------------------------
< 4GB          1-2
4-8GB          2-4
8-12GB         4-8
12-16GB        8-16
16-24GB        16-32
24GB+          32-64
```

### Usage Examples

**Basic auto-tuning:**
```bash
python tools/train_auto.py configs/detr/detr_r50_8xb2-150e_coco.py --auto-batch
```

**Manual override (if needed):**
```bash
python tools/train_auto.py configs/detr/detr_r50_8xb2-150e_coco.py --batch-size 16
```

**Check what will be used:**
```bash
python tools/train_auto.py configs/detr/detr_r50_8xb2-150e_coco.py --auto-batch --help
# The script will log the detected batch size before training
```

### When to Use

✅ **Use `--auto-batch` when:**
- You want to maximize GPU utilization
- Running on different GPU types (3090, A100, V100, etc.)
- Testing a new model and unsure about batch size
- Training on cloud instances with varying GPU memory

❌ **Don't use when:**
- You have carefully tuned hyperparameters for a specific batch size
- Reproducing published results that specify exact batch size
- Running ablation studies where batch size must be constant

## Automatic Learning Rate Scaling

### Overview

The `--auto-scale-lr` flag automatically scales learning rate when batch size or number of GPUs changes, following the **linear scaling rule** from Goyal et al. (2017).

### The Linear Scaling Rule

**Formula:** `new_lr = base_lr × (new_total_batch / base_total_batch)`

**Example:**
```
Base config: LR=0.001, batch=8, 1 GPU → total_batch=8
New setup:   LR=?,     batch=16, 4 GPUs → total_batch=64

new_lr = 0.001 × (64/8) = 0.008
```

### Usage Examples

**Basic LR scaling:**
```bash
python tools/train_auto.py configs/detr/detr_r50_8xb2-150e_coco.py --auto-scale-lr
```

**With automatic batch size:**
```bash
python tools/train_auto.py configs/detr/detr_r50_8xb2-150e_coco.py \
    --auto-batch --auto-scale-lr
```

**Multi-GPU with LR scaling:**
```bash
python -m torch.distributed.launch --nproc_per_node=8 \
    tools/train_auto.py configs/detr/detr_r50_8xb2-150e_coco.py \
    --batch-size 16 --auto-scale-lr
# LR will be scaled 8x for 8 GPUs
```

**Manual LR override:**
```bash
python tools/train_auto.py configs/detr/detr_r50_8xb2-150e_coco.py --lr 0.0001
```

### Warmup Adjustment

When learning rate is scaled up, warmup is automatically increased to ensure stable training:

```python
# Default warmup: 500 iterations for base LR
# Scaled LR 4x → warmup increased to ~2000 iterations
# Scaled LR 8x → warmup increased to ~4000 iterations (capped at 5000)
```

### When to Use

✅ **Use `--auto-scale-lr` when:**
- Training on multiple GPUs
- Changed batch size from config default
- Using `--auto-batch` flag
- Scaling to larger infrastructure

❌ **Don't use when:**
- Using very small batch sizes (< 4)
- Have custom LR schedule that's batch-size specific
- Reproducing results with exact hyperparameters

## Simplified CLI Usage

### Basic Commands

**Simplest possible:**
```bash
python tools/train_auto.py configs/your_model.py
```

**With all auto features:**
```bash
python tools/train_auto.py configs/your_model.py \
    --auto-batch \
    --auto-scale-lr \
    --amp
```

**Multi-GPU:**
```bash
python -m torch.distributed.launch --nproc_per_node=4 \
    tools/train_auto.py configs/your_model.py \
    --auto-batch \
    --auto-scale-lr
```

**With W&B logging:**
```bash
python tools/train_auto.py configs/your_model.py \
    --auto-batch \
    --auto-scale-lr \
    --wandb-project my_detection_project \
    --wandb-name experiment_1
```

### All Available Options

```bash
python tools/train_auto.py --help

# Key options:
--auto-batch              # Automatic batch size tuning
--batch-size INT          # Manual batch size override
--auto-scale-lr           # Automatic LR scaling
--lr FLOAT                # Manual LR override
--amp                     # Enable automatic mixed precision
--wandb-project NAME      # Enable W&B logging
--work-dir PATH           # Custom work directory
--seed INT                # Random seed
--resume PATH             # Resume from checkpoint
--cfg-options KEY=VALUE   # Additional config overrides
```

## Programmatic API

### Using Auto-Tune Functions

```python
from mmdet.wrappers import (
    estimate_batch_size,
    auto_scale_lr,
    suggest_training_config,
    get_gpu_info
)

# Get GPU information
num_gpus, memory_mb = get_gpu_info()
print(f"GPUs: {num_gpus}, Memory: {memory_mb}MB")

# Estimate optimal batch size
batch_size = estimate_batch_size(
    model_size='medium',  # 'tiny', 'small', 'medium', 'large', 'xlarge'
    gpu_memory_mb=memory_mb,
    image_size=(800, 1333),
    conservative=True
)
print(f"Recommended batch size: {batch_size}")

# Scale learning rate
lr_config = auto_scale_lr(
    base_lr=0.0001,
    base_batch_size=8,
    new_batch_size=16,
    num_gpus=4,
    scaling_rule='linear'  # 'linear', 'sqrt', 'none'
)
print(f"Scaled LR: {lr_config['lr']}")
print(f"Warmup iters: {lr_config['warmup_iters']}")

# Get complete suggestions
suggestions = suggest_training_config(
    config_path='configs/detr/detr_r50_8xb2-150e_coco.py',
    num_gpus=4,
    auto_batch=True,
    auto_lr=True
)
print(f"Suggested batch size: {suggestions['batch_size']}")
print(f"Suggested LR: {suggestions['lr']}")
print(f"Use AMP: {suggestions['amp']}")
```

### Integration with Existing Code

```python
from mmdet.wrappers import HydraWrapper, suggest_training_config

# Get optimal configuration
suggestions = suggest_training_config(
    'configs/detr/detr_r50_8xb2-150e_coco.py',
    num_gpus=4
)

# Create wrapper
wrapper = HydraWrapper(
    config_path='configs/detr/detr_r50_8xb2-150e_coco.py',
    work_dir='./work_dirs/auto_tuned'
)

# Load and apply suggestions
config = wrapper.load_config()
config.train_dataloader.batch_size = suggestions['batch_size']
config.optim_wrapper.optimizer.lr = suggestions['lr']

if suggestions['amp']:
    config.optim_wrapper.type = 'AmpOptimWrapper'
    config.optim_wrapper.loss_scale = 'dynamic'

# Train
results = wrapper.train(config)
```

## Best Practices

### 1. Start with Auto-Tuning

```bash
# First run: let the system optimize
python tools/train_auto.py configs/model.py --auto-batch --auto-scale-lr

# Review the logs to see what was chosen
# Adjust manually if needed in subsequent runs
```

### 2. Multi-GPU Training

```bash
# Always use auto-scale-lr with multi-GPU
python -m torch.distributed.launch --nproc_per_node=4 \
    tools/train_auto.py configs/model.py --auto-scale-lr
```

### 3. Memory-Constrained Environments

```bash
# Use smaller batch size but still scale LR
python tools/train_auto.py configs/model.py \
    --batch-size 4 \
    --auto-scale-lr \
    --amp  # AMP can help with memory
```

### 4. Reproducibility

```bash
# Fix batch size and LR for reproducible results
python tools/train_auto.py configs/model.py \
    --batch-size 8 \
    --lr 0.0001 \
    --seed 42
```

### 5. Experiment Tracking

```bash
# Use W&B to track different configurations
python tools/train_auto.py configs/model.py \
    --auto-batch \
    --auto-scale-lr \
    --wandb-project detection_experiments \
    --wandb-name auto_tuned_run_1
```

## Advanced Usage

### Custom Model Sizes

```python
from mmdet.wrappers import estimate_batch_size

# For very large models
batch_size = estimate_batch_size(
    model_size='xlarge',
    gpu_memory_mb=40000,  # A100 40GB
    image_size=(1024, 1024),
    conservative=True
)

# For tiny models
batch_size = estimate_batch_size(
    model_size='tiny',
    gpu_memory_mb=8000,
    image_size=(416, 416),
    conservative=False  # More aggressive
)
```

### Square Root LR Scaling

For very large batch sizes (>1024), consider sqrt scaling:

```python
from mmdet.wrappers import auto_scale_lr

lr_config = auto_scale_lr(
    base_lr=0.0001,
    base_batch_size=8,
    new_batch_size=128,
    num_gpus=32,
    scaling_rule='sqrt'  # Less aggressive than linear
)
```

### Batch Size Search

```bash
# Try different batch sizes
for bs in 8 16 32 64; do
    python tools/train_auto.py configs/model.py \
        --batch-size $bs \
        --auto-scale-lr \
        --work-dir ./work_dirs/batch_${bs} \
        --wandb-name batch_size_${bs}
done
```

### Memory Profiling

```python
import torch
from mmdet.wrappers import get_gpu_info

# Before training
num_gpus, free_memory = get_gpu_info()
print(f"Free memory: {free_memory}MB")

# Monitor during training
torch.cuda.reset_peak_memory_stats()
# ... training code ...
peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
print(f"Peak memory: {peak_memory}MB")
```

## Troubleshooting

### Out of Memory (OOM)

If you still get OOM with `--auto-batch`:

```bash
# Reduce batch size manually
python tools/train_auto.py configs/model.py --batch-size 4

# Enable AMP to reduce memory
python tools/train_auto.py configs/model.py --auto-batch --amp

# Use gradient accumulation (in config)
# Set train_dataloader.batch_size to smaller value
# and increase num_iters in scheduler
```

### Learning Rate Too High/Low

```bash
# Manually tune LR
python tools/train_auto.py configs/model.py --lr 0.00005

# Use sqrt scaling for very large batches
# (requires programmatic API)
```

### Slow Training

```bash
# Enable AMP for faster training
python tools/train_auto.py configs/model.py --amp

# Increase batch size if memory allows
python tools/train_auto.py configs/model.py --batch-size 32

# Use multiple GPUs
python -m torch.distributed.launch --nproc_per_node=4 \
    tools/train_auto.py configs/model.py --auto-batch
```

## References

- **Linear Scaling Rule:** Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (2017)
  - https://arxiv.org/abs/1706.02677
- **Large Batch Training:** You et al., "Large Batch Training of Convolutional Networks" (2017)
  - https://arxiv.org/abs/1708.03888
- **Learning Rate Warmup:** He et al., "Bag of Tricks for Image Classification with CNNs" (2018)
  - https://arxiv.org/abs/1812.01187

## Examples

See `examples/custom_dataset/` for complete examples using auto-tuning:
- `1_basic_custom_dataset.py` - Basic usage with auto features
- `3_multi_gpu_training.py` - Multi-GPU with auto LR scaling
- `4_hyperparameter_tuning.py` - Ray Tune with auto-tuning

## Need Help?

- **Documentation:** See `docs/WRAPPER_GUIDE.md` for general wrapper usage
- **Quick Start:** See `docs/QUICKSTART_WRAPPER.md` for 5-minute guide
- **Multi-GPU:** See `docs/MULTI_GPU_AND_EXPORT.md` for distributed training
- **GitHub Issues:** Open an issue for questions or problems
