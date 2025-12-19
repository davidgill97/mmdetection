# MM Wrapper System Examples

This directory contains example configurations and usage demonstrations for the MM wrapper system.

## Files

- **`hydra_configs/`**: Example Hydra configuration files
  - `base_config.yaml`: Base Hydra configuration template
  - `detr_coco.yaml`: Example config for DETR training on COCO
  - `mmseg_pspnet.yaml`: Example config for MMSegmentation PSPNet

- **`ray_tune_search_space.yaml`**: Example search space for Ray Tune hyperparameter optimization

- **`wrapper_usage_examples.py`**: Demonstration script showing programmatic usage of wrappers

## Running Examples

### 1. View Wrapper Usage Examples

Run the example script to see how to use wrappers programmatically:

```bash
cd /home/runner/work/mmdetection/mmdetection
python examples/wrapper_usage_examples.py
```

This script demonstrates:
- Loading MM configs with HydraWrapper
- Using Hydra YAML configs
- Config conversion utilities
- Multi-library support with ConfigManager
- Setting up Ray Tune

### 2. Train with Hydra Config

```bash
# Using Hydra config
python tools/train_hydra.py \
    --config-path ../examples/hydra_configs \
    --config-name deformable_detr_mnm

# Using MM config directly
python tools/train_hydra.py \
    --mm-config configs/detr/detr_r50_8xb2-150e_coco.py \
    --work-dir ./work_dirs/test_hydra
```

### 3. Hyperparameter Tuning with Ray

```bash
# Single training run
python tools/train_ray.py \
    --config configs/detr/detr_r50_8xb2-150e_coco.py \
    --single-run \
    --work-dir ./work_dirs/test_ray

# Hyperparameter tuning with custom search space
python tools/train_ray.py \
    --config configs/detr/detr_r50_8xb2-150e_coco.py \
    --search-space examples/ray_tune_search_space.yaml \
    --num-samples 10 \
    --gpus-per-trial 1
```

## Creating Your Own Configs

### Hydra Config

Create a new YAML file in `hydra_configs/`:

```yaml
# my_experiment.yaml
defaults:
  - base_config
  - _self_

mm_config: ../../configs/path/to/your/config.py

wandb:
  project: my_project
  name: my_experiment

optimizer:
  lr: 0.0001

data:
  train_batch_size: 16
```

### Ray Tune Search Space

Create a YAML file defining your search space:

```yaml
# my_search_space.yaml

# Learning rate
optim_wrapper.optimizer.lr:
  type: loguniform
  min: 0.00001
  max: 0.001

# Batch size
train_dataloader.batch_size:
  type: choice
  choices: [8, 16, 32]
```

## Notes

- Ensure you have the required dependencies installed (see `requirements/wrapper.txt`)
- Hydra configs are relative to the config directory
- MM config paths in Hydra configs should be relative to the Hydra config file
- Ray Tune results are saved in the specified `work_dir`

## Documentation

For detailed documentation, see `docs/WRAPPER_GUIDE.md`.
