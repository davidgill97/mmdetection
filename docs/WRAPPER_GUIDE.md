# MM Libraries Wrapper System

This wrapper system provides integration between MM libraries (mmdetection, mmsegmentation, etc.) and modern ML frameworks like Hydra and Ray Tune. It enables hierarchical configuration management, easy parameter overrides, wandb logging, and scalable hyperparameter tuning.

## Features

- **Hydra Integration**: Hierarchical configuration composition with easy overrides
- **Ray Tune Integration**: Scalable hyperparameter optimization with multiple search algorithms
- **Wandb Logging**: Integrated experiment tracking and visualization
- **Config Compatibility**: Seamless integration with existing MM config system
- **Extensible Design**: Easy to extend to other MM libraries (mmseg, mmcls, etc.)
- **Unified Interface**: Consistent API across different frameworks

## Installation

### Basic Requirements
```bash
pip install mmengine mmdet
```

### Hydra Support
```bash
pip install hydra-core omegaconf
```

### Ray Tune Support
```bash
pip install ray[tune] hyperopt
```

### All Features
```bash
pip install hydra-core omegaconf ray[tune] hyperopt wandb
```

## Quick Start

### 1. Training with Hydra

Hydra allows for hierarchical configuration management with easy parameter overrides.

#### Using Hydra Config
```bash
python tools/train_hydra.py \
    --config-path ../examples/hydra_configs \
    --config-name deformable_detr_mnm
```

#### With Overrides
```bash
python tools/train_hydra.py \
    --config-path ../examples/hydra_configs \
    --config-name deformable_detr_mnm \
    --overrides \
        wandb.project=my_project \
        optimizer.lr=0.0001 \
        data.train_batch_size=16
```

#### Using MM Config Directly
```bash
python tools/train_hydra.py \
    --mm-config configs/detr/detr_r50_8xb2-150e_coco.py \
    --work-dir ./my_work_dir
```

### 2. Hyperparameter Tuning with Ray Tune

Ray Tune enables scalable hyperparameter optimization with various search algorithms.

#### Basic Tuning
```bash
python tools/train_ray.py \
    --config configs/detr/detr_r50_8xb2-150e_coco.py \
    --num-samples 20 \
    --metric coco/bbox_mAP \
    --gpus-per-trial 1
```

#### With Custom Search Space
```bash
python tools/train_ray.py \
    --config configs/detr/detr_r50_8xb2-150e_coco.py \
    --search-space examples/ray_tune_search_space.yaml \
    --num-samples 50 \
    --scheduler asha \
    --search-alg hyperopt
```

#### Single Training Run (No Tuning)
```bash
python tools/train_ray.py \
    --config configs/detr/detr_r50_8xb2-150e_coco.py \
    --single-run \
    --work-dir ./ray_work_dir
```

## Configuration System

### Hydra Configuration Structure

Hydra configs are YAML files that can compose and override MM configs:

```yaml
# examples/hydra_configs/my_config.yaml
defaults:
  - base_config
  - _self_

# Reference to MM config
mm_config: ../../configs/detr/detr_r50_8xb2-150e_coco.py

# Work directory with timestamp
work_dir: ./work_dirs/my_exp/${now:%Y%m%d_%H%M%S}

# Wandb configuration
wandb:
  project: my_project
  name: my_experiment
  tags: [detr, custom]

# Override specific parameters
optimizer:
  lr: 0.0002
  weight_decay: 0.0001

data:
  train_batch_size: 16
  val_batch_size: 16

model:
  num_classes: 10
```

### Ray Tune Search Space

Define hyperparameter search spaces in YAML:

```yaml
# examples/my_search_space.yaml

# Learning rate - log uniform distribution
optim_wrapper.optimizer.lr:
  type: loguniform
  min: 0.00001
  max: 0.001

# Batch size - discrete choices
train_dataloader.batch_size:
  type: choice
  choices: [8, 16, 32]

# Weight decay
optim_wrapper.optimizer.weight_decay:
  type: loguniform
  min: 0.000001
  max: 0.001
```

Supported types:
- `uniform`: Uniform distribution between min and max
- `loguniform`: Log-uniform distribution
- `choice`: Discrete choices from a list
- `randint`: Random integer between min and max

## Wandb Integration

Both wrappers support wandb logging. Configure in Hydra config:

```yaml
wandb:
  project: my_project
  entity: my_team
  name: experiment_name
  tags: [tag1, tag2]
  group: experiment_group
  notes: "Experiment description"
```

Or use existing MM config wandb settings:

```python
# In MM config
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs=dict(
             project='my_project',
             name='my_experiment',
             tags=['detection', 'detr']
         ))
]
```

## Wrapper API

### HydraWrapper

```python
from mmdet.wrappers import HydraWrapper

# Initialize with Hydra config
wrapper = HydraWrapper(
    config_path='./examples/hydra_configs',
    config_name='deformable_detr_mnm',
    work_dir='./work_dirs',
    overrides=['optimizer.lr=0.0001']
)

# Load and merge configs
config = wrapper.load_config()

# Train
results = wrapper.train(config)
```

### RayWrapper

```python
from mmdet.wrappers import RayWrapper
from ray import tune

# Initialize with search space
search_space = {
    'optim_wrapper.optimizer.lr': tune.loguniform(1e-5, 1e-3),
    'train_dataloader.batch_size': tune.choice([8, 16, 32])
}

wrapper = RayWrapper(
    config_path='configs/detr/detr_r50_8xb2-150e_coco.py',
    search_space=search_space,
    num_samples=20,
    metric='coco/bbox_mAP',
    mode='max'
)

# Run hyperparameter tuning
results = wrapper.train()
print(f"Best config: {results['best_config']}")
print(f"Best metrics: {results['best_metrics']}")
```

## Extending to Other MM Libraries

The wrapper system is designed to work with any MM library. Here's how to extend it:

1. **Use ConfigManager**:
```python
from mmdet.wrappers.config_utils import ConfigManager

# Initialize for mmsegmentation
config_manager = ConfigManager(library='mmseg')
config = config_manager.load_config('path/to/mmseg/config.py')
```

2. **Use the same wrappers**:
```python
from mmdet.wrappers import HydraWrapper

# Works with any MM library config
wrapper = HydraWrapper(
    config_path='path/to/mmseg/config.py',
    config_name=None  # Use MM config directly
)
```

3. **Create library-specific configs**:
```yaml
# examples/hydra_configs/mmseg_config.yaml
mm_config: path/to/mmseg/pspnet_config.py
wandb:
  project: mmsegmentation
# ... other settings
```

## Advanced Usage

### Custom Search Algorithms

```python
from ray.tune.search.bayesopt import BayesOptSearch

wrapper = RayWrapper(
    config_path='...',
    search_space={...},
    search_alg='bayesopt',  # or implement custom
    num_samples=50
)
```

### Population Based Training

```python
wrapper = RayWrapper(
    config_path='...',
    search_space={...},
    scheduler='pbt',  # Population Based Training
    num_samples=50
)
```

### Multi-GPU Training

```bash
python tools/train_ray.py \
    --config ... \
    --gpus-per-trial 2 \
    --cpus-per-trial 4
```

## Best Practices

1. **Start with Hydra**: Use Hydra for configuration management even without Ray Tune
2. **Modular Configs**: Create modular Hydra configs that can be composed
3. **Version Control**: Keep search space definitions in version control
4. **Wandb Sweeps**: Use wandb sweeps for additional experiment tracking
5. **Checkpointing**: Ray Tune automatically handles checkpointing
6. **Resource Management**: Set appropriate `gpus_per_trial` and `cpus_per_trial`

## Troubleshooting

### Hydra Not Found
```bash
pip install hydra-core omegaconf
```

### Ray Not Found
```bash
pip install ray[tune]
```

### Config Loading Errors
- Ensure MM config paths are correct (absolute or relative to Hydra config)
- Check for circular dependencies in config composition

### Memory Issues
- Reduce `gpus_per_trial` or `num_samples`
- Use ASHA scheduler for early stopping

## Examples

See the `examples/` directory for:
- `hydra_configs/`: Example Hydra configurations
- `ray_tune_search_space.yaml`: Example search space definition

## License

This wrapper system is part of mmdetection and follows the same Apache 2.0 License.
