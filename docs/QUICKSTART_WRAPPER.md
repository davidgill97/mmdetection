# Quick Start Guide: MM Wrapper System

This guide will help you quickly get started with the Hydra and Ray wrapper system for mmdetection.

## Prerequisites

```bash
# Basic installation (already done if mmdetection is installed)
pip install mmengine mmdet

# Optional: For Hydra support
pip install hydra-core omegaconf

# Optional: For Ray Tune support
pip install ray[tune] hyperopt

# Optional: For wandb logging
pip install wandb
```

## 5-Minute Quick Start

### 1. Verify Installation

Run the example script to verify everything is working:

```bash
python examples/wrapper_usage_examples.py
```

### 2. Basic Training with MM Config

Use the Hydra wrapper to train with existing MM configs:

```bash
python tools/train_hydra.py \
    --mm-config configs/neurocle/deformable_detr/deformable_detr_r50_1xb2-15e_mnm.py \
    --work-dir ./work_dirs/my_first_training
```

### 3. Training with Hydra Config (Optional)

If you have Hydra installed, you can use hierarchical configs:

```bash
# Install Hydra first
pip install hydra-core omegaconf

# Train with Hydra config
python tools/train_hydra.py \
    --config-path ../examples/hydra_configs \
    --config-name deformable_detr_mnm
```

### 4. Single Training with Ray (Optional)

If you have Ray installed, you can use the Ray wrapper:

```bash
# Install Ray first
pip install ray[tune]

# Single training run
python tools/train_ray.py \
    --config configs/neurocle/deformable_detr/deformable_detr_r50_1xb2-15e_mnm.py \
    --single-run \
    --work-dir ./work_dirs/ray_training
```

## Key Features at a Glance

### ✨ Hydra Wrapper
- ✅ Works with existing MM configs (no Hydra installation needed)
- ✅ Optional Hydra config support for hierarchical composition
- ✅ Easy parameter overrides via command line
- ✅ Integrated wandb logging

### ✨ Ray Wrapper
- ✅ Single training runs without tuning
- ✅ Scalable hyperparameter optimization
- ✅ Multiple search algorithms (HyperOpt, Random, etc.)
- ✅ Multiple schedulers (ASHA, PBT)
- ✅ Integrated wandb logging

## Common Use Cases

### Use Case 1: Quick Experiment with Different Parameters

```bash
python tools/train_hydra.py \
    --mm-config configs/neurocle/deformable_detr/deformable_detr_r50_1xb2-15e_mnm.py \
    --overrides \
        optimizer.lr=0.0001 \
        data.train_batch_size=16 \
        wandb.project=my_project
```

### Use Case 2: Hyperparameter Tuning

```bash
python tools/train_ray.py \
    --config configs/neurocle/deformable_detr/deformable_detr_r50_1xb2-15e_mnm.py \
    --search-space examples/ray_tune_search_space.yaml \
    --num-samples 20 \
    --gpus-per-trial 1
```

### Use Case 3: Programmatic Usage

```python
from mmdet.wrappers import HydraWrapper

# Initialize wrapper
wrapper = HydraWrapper(
    config_path='configs/path/to/config.py',
    work_dir='./my_experiments'
)

# Load and customize config
config = wrapper.load_config()
config = wrapper.merge_config({'optimizer.lr': 0.0001})

# Train
results = wrapper.train(config)
```

## Next Steps

1. **Read the full documentation**: [docs/WRAPPER_GUIDE.md](../docs/WRAPPER_GUIDE.md)
2. **Explore examples**: Check out [examples/](../examples/) directory
3. **Create your own configs**: Use templates in [examples/hydra_configs/](../examples/hydra_configs/)

## Troubleshooting

### "Hydra is not installed"
- This is normal if you're using MM configs directly
- Install Hydra only if you want to use hierarchical configs: `pip install hydra-core omegaconf`

### "Ray not installed"
- Install Ray only if you want hyperparameter tuning: `pip install ray[tune]`

### Config Loading Errors
- Ensure config paths are correct
- For Hydra configs, paths should be relative to the config directory
- For MM configs, use absolute paths or relative to current directory

## Need Help?

- Full documentation: [docs/WRAPPER_GUIDE.md](../docs/WRAPPER_GUIDE.md)
- Examples: [examples/README.md](../examples/README.md)
- Issues: Open an issue on GitHub

---

**Ready to go?** Start with the basic training command and explore from there! 🚀
