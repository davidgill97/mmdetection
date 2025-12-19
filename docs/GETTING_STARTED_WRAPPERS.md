# Getting Started with MM Wrapper System

Welcome! This guide will walk you through using the new Hydra and Ray wrapper system for mmdetection.

## 📋 Table of Contents

1. [What is this?](#what-is-this)
2. [Installation](#installation)
3. [Quick Examples](#quick-examples)
4. [Step-by-Step Tutorial](#step-by-step-tutorial)
5. [Common Workflows](#common-workflows)
6. [Next Steps](#next-steps)

## What is this?

The MM wrapper system adds powerful features to mmdetection:

- **🔧 Easy Configuration**: Use Hydra for hierarchical configs with simple overrides
- **⚡ Hyperparameter Tuning**: Scale up with Ray Tune for automatic optimization
- **📊 Experiment Tracking**: Built-in wandb integration
- **🔄 Flexibility**: Works with existing MM configs OR Hydra configs
- **🌐 Extensible**: Use with other MM libraries (mmseg, mmcls, etc.)

**No breaking changes** - your existing workflows still work!

## Installation

### Basic Installation (Already Done)

If mmdetection is installed, you already have the core components:
```bash
# These are already installed
pip install mmengine mmdet
```

### Optional: Add Hydra Support

```bash
pip install hydra-core omegaconf
```

### Optional: Add Ray Tune Support

```bash
pip install ray[tune] hyperopt
```

### Optional: All Features

```bash
pip install -r requirements/wrapper.txt
```

## Quick Examples

### Example 1: Train with Existing MM Config

No new dependencies needed! Just use the new training script:

```bash
python tools/train_hydra.py \
    --mm-config configs/neurocle/deformable_detr/deformable_detr_r50_1xb2-15e_mnm.py \
    --work-dir ./my_training
```

**That's it!** You're using the wrapper system with your existing config.

### Example 2: Override Parameters Easily

```bash
python tools/train_hydra.py \
    --mm-config configs/neurocle/deformable_detr/deformable_detr_r50_1xb2-15e_mnm.py \
    --overrides \
        optimizer.lr=0.0001 \
        data.train_batch_size=16 \
        wandb.project=my_awesome_project
```

### Example 3: Hyperparameter Tuning (Requires Ray)

```bash
# Install Ray first
pip install ray[tune]

# Run tuning
python tools/train_ray.py \
    --config configs/neurocle/deformable_detr/deformable_detr_r50_1xb2-15e_mnm.py \
    --num-samples 10 \
    --gpus-per-trial 1
```

## Step-by-Step Tutorial

### Tutorial 1: Basic Training with Wrappers

**Goal**: Train a model using the wrapper system

**Steps**:

1. **Verify installation**:
```bash
python -c "from mmdet.wrappers import HydraWrapper; print('✓ Ready!')"
```

2. **Run training**:
```bash
python tools/train_hydra.py \
    --mm-config configs/neurocle/deformable_detr/deformable_detr_r50_1xb2-15e_mnm.py \
    --work-dir ./tutorial_1
```

3. **Check results**:
```bash
ls -la ./tutorial_1/
# You'll see logs, checkpoints, etc.
```

**Done!** You've trained using the wrapper system.

### Tutorial 2: Using Hydra Configs

**Goal**: Create and use a Hydra config for easier parameter management

**Steps**:

1. **Install Hydra**:
```bash
pip install hydra-core omegaconf
```

2. **Use example config**:
```bash
python tools/train_hydra.py \
    --config-path ../examples/hydra_configs \
    --config-name deformable_detr_mnm
```

3. **Override parameters**:
```bash
python tools/train_hydra.py \
    --config-path ../examples/hydra_configs \
    --config-name deformable_detr_mnm \
    --overrides \
        optimizer.lr=0.0002 \
        wandb.name=my_experiment
```

**Done!** You're using hierarchical configs with Hydra.

### Tutorial 3: Hyperparameter Tuning

**Goal**: Find the best hyperparameters automatically

**Steps**:

1. **Install Ray**:
```bash
pip install ray[tune] hyperopt
```

2. **Review search space** (optional):
```bash
cat examples/ray_tune_search_space.yaml
```

3. **Run tuning**:
```bash
python tools/train_ray.py \
    --config configs/neurocle/deformable_detr/deformable_detr_r50_1xb2-15e_mnm.py \
    --search-space examples/ray_tune_search_space.yaml \
    --num-samples 5 \
    --gpus-per-trial 1 \
    --work-dir ./ray_results
```

4. **Check results**:
```bash
# Results are in ./ray_results/
# Best config is printed at the end
```

**Done!** Ray Tune found the best hyperparameters for you.

## Common Workflows

### Workflow 1: Quick Experiment

You want to try different learning rates quickly:

```bash
# Try LR 1e-4
python tools/train_hydra.py \
    --mm-config configs/neurocle/deformable_detr/deformable_detr_r50_1xb2-15e_mnm.py \
    --overrides optimizer.lr=0.0001 \
    --work-dir ./exp_lr_1e4

# Try LR 2e-4
python tools/train_hydra.py \
    --mm-config configs/neurocle/deformable_detr/deformable_detr_r50_1xb2-15e_mnm.py \
    --overrides optimizer.lr=0.0002 \
    --work-dir ./exp_lr_2e4
```

### Workflow 2: Systematic Hyperparameter Search

You want to find the best combination of LR, batch size, and weight decay:

1. **Create search space** (`my_search.yaml`):
```yaml
optim_wrapper.optimizer.lr:
  type: loguniform
  min: 0.00001
  max: 0.001

train_dataloader.batch_size:
  type: choice
  choices: [8, 16, 32]

optim_wrapper.optimizer.weight_decay:
  type: loguniform
  min: 0.000001
  max: 0.001
```

2. **Run search**:
```bash
python tools/train_ray.py \
    --config configs/neurocle/deformable_detr/deformable_detr_r50_1xb2-15e_mnm.py \
    --search-space my_search.yaml \
    --num-samples 20 \
    --scheduler asha
```

### Workflow 3: Programmatic Usage

You want full control in Python:

```python
from mmdet.wrappers import HydraWrapper

# Initialize
wrapper = HydraWrapper(
    config_path='configs/neurocle/deformable_detr/deformable_detr_r50_1xb2-15e_mnm.py',
    work_dir='./my_experiment'
)

# Load and customize
config = wrapper.load_config()
config = wrapper.merge_config({
    'optimizer': {'lr': 0.0001},
    'train_dataloader': {'batch_size': 16}
})

# Train
results = wrapper.train(config)
print(f"Training complete! mAP: {results.get('coco/bbox_mAP', 'N/A')}")
```

## Next Steps

### Learn More

- **Quick Start**: [docs/QUICKSTART_WRAPPER.md](./docs/QUICKSTART_WRAPPER.md)
- **Full Guide**: [docs/WRAPPER_GUIDE.md](./docs/WRAPPER_GUIDE.md)
- **Extension Guide**: [docs/EXTENDING_TO_OTHER_MM_LIBS.md](./docs/EXTENDING_TO_OTHER_MM_LIBS.md)

### Run Examples

```bash
# See all examples
python examples/wrapper_usage_examples.py

# Run tests
python tests/test_wrappers_integration.py
```

### Try Advanced Features

1. **Create your own Hydra config**:
   - Copy `examples/hydra_configs/base_config.yaml`
   - Modify for your use case
   - Reference your MM config

2. **Design custom search space**:
   - Copy `examples/ray_tune_search_space.yaml`
   - Add your hyperparameters
   - Run Ray Tune

3. **Use with other MM libraries**:
   - Same wrappers work with mmseg, mmcls, etc.
   - See extension guide for details

### Get Help

- **Documentation**: Check `docs/` directory
- **Examples**: Check `examples/` directory  
- **Tests**: Run `tests/test_wrappers_integration.py`
- **Issues**: Open a GitHub issue

## Tips & Tricks

### Tip 1: Start Simple
Begin with MM configs and basic wrapper usage. Add Hydra and Ray later.

### Tip 2: Use Work Directories
Always specify `--work-dir` to organize experiments:
```bash
--work-dir ./experiments/exp_$(date +%Y%m%d_%H%M%S)
```

### Tip 3: Leverage Wandb
Configure wandb in your overrides:
```bash
--overrides \
    wandb.project=my_project \
    wandb.name=exp_001 \
    wandb.tags='[detr,finetuning]'
```

### Tip 4: Test Before Full Training
Use smaller datasets or fewer epochs for testing:
```bash
--overrides \
    train_dataloader.batch_size=2 \
    max_epochs=1
```

### Tip 5: Save Configurations
Save successful configs for reuse:
```bash
# Hydra configs are automatically saved
# Check your work_dir/.hydra/ directory
```

## Troubleshooting

### "Hydra is not installed"
- **Solution**: This is expected! HydraWrapper works with MM configs without Hydra
- **Or**: Install Hydra if you want hierarchical configs: `pip install hydra-core`

### "Ray not installed"
- **Solution**: Install Ray only if you need hyperparameter tuning: `pip install ray[tune]`

### Config not found
- **Check**: Config paths are correct
- **Try**: Use absolute paths
- **For Hydra**: Paths are relative to the config directory

### Out of memory
- **Reduce**: Batch size with `--overrides data.train_batch_size=4`
- **Or**: Use fewer GPUs per trial: `--gpus-per-trial 0.5`

## Congratulations! 🎉

You now know how to use the MM wrapper system! Start with simple examples and gradually explore more advanced features.

**Happy training!** 🚀
