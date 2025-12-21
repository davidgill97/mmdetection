# Multi-GPU Training and Model Export Guide

This guide covers advanced topics for production deployments: multi-GPU training and model export.

## Multi-GPU Training

The wrapper system fully supports distributed training across multiple GPUs using PyTorch's DistributedDataParallel.

### Using Hydra Wrapper

#### Method 1: PyTorch Distributed Launch

```bash
# Train on 4 GPUs using torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    tools/train_hydra.py \
    --mm-config configs/detr/detr_r50_8xb2-150e_coco.py \
    --launcher pytorch
```

#### Method 2: Using Config File

Set the launcher in your Hydra config:

```yaml
# examples/hydra_configs/detr_multi_gpu.yaml
mm_config: ../../configs/detr/detr_r50_8xb2-150e_coco.py

# Enable distributed training
launcher: pytorch

# Adjust batch size per GPU
data:
  train_batch_size: 2  # Per GPU batch size
  val_batch_size: 2
  
# With 4 GPUs, effective batch size = 2 * 4 = 8
```

Then launch with:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    tools/train_hydra.py \
    --config-path ../examples/hydra_configs \
    --config-name detr_multi_gpu
```

### Using Ray Tune with Multiple GPUs

Ray Tune automatically handles multi-GPU allocation:

```bash
# Allocate 1 GPU per trial, run 4 trials in parallel on 4 GPUs
python tools/train_ray.py \
    --config configs/detr/detr_r50_8xb2-150e_coco.py \
    --num-samples 20 \
    --gpus-per-trial 1 \
    --cpus-per-trial 4

# Or use 2 GPUs per trial, run 2 trials in parallel on 4 GPUs
python tools/train_ray.py \
    --config configs/detr/detr_r50_8xb2-150e_coco.py \
    --num-samples 10 \
    --gpus-per-trial 2 \
    --cpus-per-trial 8 \
    --launcher pytorch
```

### Best Practices for Multi-GPU Training

1. **Batch Size Scaling**: When using N GPUs, the effective batch size = `batch_size_per_gpu * N`
   - Adjust learning rate proportionally: `lr_new = lr_base * N`
   - Or use gradient accumulation if memory is limited

2. **Worker Allocation**: Set `num_workers` based on available CPUs
   - Rule of thumb: `num_workers = min(cpus_per_gpu * 2, 8)`
   - Example for 4 GPUs: `num_workers = 4` per GPU

3. **Synchronization**: The wrapper handles batch norm synchronization automatically when using `launcher=pytorch`

4. **Checkpointing**: Checkpoints are saved on rank 0 only to avoid conflicts

### Programmatic Multi-GPU Training

```python
from mmdet.wrappers import HydraWrapper

wrapper = HydraWrapper(
    config_path='configs/detr/detr_r50_8xb2-150e_coco.py',
    work_dir='./work_dirs/multi_gpu_training'
)

config = wrapper.load_config()

# Configure for distributed training
config.launcher = 'pytorch'
config.train_dataloader.batch_size = 2  # Per GPU

# Launch with torch.distributed
# (This should be called with torch.distributed.launch)
results = wrapper.train(config)
```

## Model Export for Deployment

The wrapper system integrates with MMDeploy for exporting models to production formats.

### Prerequisites

```bash
# Install MMDeploy
pip install mmdeploy mmdeploy-runtime

# Install backend dependencies
# For ONNX Runtime
pip install onnxruntime-gpu

# For TensorRT (requires CUDA)
pip install tensorrt
```

### Exporting to ONNX

After training, export your model:

```python
from mmdet.wrappers import HydraWrapper
from mmdeploy.apis import torch2onnx

# Train the model
wrapper = HydraWrapper(
    config_path='configs/detr/detr_r50_8xb2-150e_coco.py',
    work_dir='./work_dirs/detr_training'
)
config = wrapper.load_config()
results = wrapper.train(config)

# Export to ONNX
work_dir = './work_dirs/detr_onnx'
checkpoint = './work_dirs/detr_training/latest.pth'

torch2onnx(
    img='demo/demo.jpg',
    work_dir=work_dir,
    save_file='end2end.onnx',
    deploy_cfg='configs/mmdeploy/detection_onnxruntime_dynamic.py',
    model_cfg=config,
    model_checkpoint=checkpoint,
    device='cuda:0'
)
```

### Exporting to TensorRT

For maximum inference speed on NVIDIA GPUs, first export to ONNX then convert to TensorRT:

```python
from mmdeploy.apis import torch2onnx, onnx2tensorrt

# Step 1: Export to ONNX
work_dir_onnx = './work_dirs/detr_onnx'
checkpoint = './work_dirs/detr_training/latest.pth'

torch2onnx(
    img='demo/demo.jpg',
    work_dir=work_dir_onnx,
    save_file='end2end.onnx',
    deploy_cfg='configs/mmdeploy/detection_onnxruntime_dynamic.py',
    model_cfg=config,
    model_checkpoint=checkpoint,
    device='cuda:0'
)

# Step 2: Convert ONNX to TensorRT
work_dir_trt = './work_dirs/detr_tensorrt'

onnx2tensorrt(
    work_dir=work_dir_trt,
    save_file='end2end.engine',
    model_id=0,
    deploy_cfg='configs/mmdeploy/detection_tensorrt_dynamic-320x320-1344x1344.py',
    onnx_model=f'{work_dir_onnx}/end2end.onnx',
    device='cuda:0'
)
```

### Using Exported Models

```python
from mmdeploy.apis import inference_model

# ONNX inference
model = inference_model(
    model_cfg='configs/detr/detr_r50_8xb2-150e_coco.py',
    deploy_cfg='configs/mmdeploy/detection_onnxruntime_dynamic.py',
    backend_files=['./work_dirs/detr_onnx/end2end.onnx'],
    device='cuda:0'
)

# Run inference
result = model(['demo/demo.jpg'])
```

### Export-Friendly Training Configuration

When training models for export, consider:

1. **Static Input Shapes**: Use fixed input sizes for better optimization
   ```yaml
   data:
     train_pipeline:
       - type: Resize
         scale: (800, 800)
         keep_ratio: false
   ```

2. **Avoid Dynamic Operations**: Some operations may not export well
   - Use standard convolutions over dynamic ones when possible
   - Avoid custom CUDA operations without ONNX support

3. **Test Export Early**: Verify exportability during development
   ```bash
   # Quick export test
   python tools/deployment/pytorch2onnx.py \
       configs/detr/detr_r50_8xb2-150e_coco.py \
       checkpoints/detr_r50.pth \
       --output-file detr_test.onnx \
       --verify
   ```

### Complete Workflow Example

```python
from mmdet.wrappers import HydraWrapper
from mmdeploy.apis import torch2onnx, onnx2tensorrt

# 1. Train with multi-GPU
wrapper = HydraWrapper(
    config_path='configs/detr/detr_r50_8xb2-150e_coco.py',
    work_dir='./work_dirs/production_training'
)

config = wrapper.load_config()
config.launcher = 'pytorch'  # Enable multi-GPU
config.train_dataloader.batch_size = 4  # Per GPU batch size

# Launch with: python -m torch.distributed.launch --nproc_per_node=4 script.py
results = wrapper.train(config)

# 2. Export best checkpoint to ONNX
best_checkpoint = './work_dirs/production_training/best_coco_bbox_mAP_epoch_150.pth'

torch2onnx(
    img='demo/demo.jpg',
    work_dir='./work_dirs/exported_model',
    save_file='model.onnx',
    deploy_cfg='configs/mmdeploy/detection_onnxruntime_dynamic.py',
    model_cfg=config,
    model_checkpoint=best_checkpoint,
    device='cuda:0'
)

# 3. (Optional) Convert ONNX to TensorRT for faster inference
onnx2tensorrt(
    work_dir='./work_dirs/exported_model',
    save_file='model.engine',
    model_id=0,
    deploy_cfg='configs/mmdeploy/detection_tensorrt_dynamic-320x320-1344x1344.py',
    onnx_model='./work_dirs/exported_model/model.onnx',
    device='cuda:0'
)

# 3. Verify exported model
from mmdeploy.apis import inference_model

model = inference_model(
    model_cfg=config,
    deploy_cfg='configs/mmdeploy/detection_onnxruntime_dynamic.py',
    backend_files=['./work_dirs/exported_model/model.onnx'],
    device='cuda:0'
)

# Test inference
results = model(['demo/demo.jpg'])
print(f"Inference successful: {len(results[0])} detections")
```

## Integration with Other MM Libraries

The export process works similarly for other MM libraries:

### MMSegmentation Export

```python
from mmdet.wrappers import HydraWrapper, ConfigManager

# Train segmentation model
manager = ConfigManager(library='mmseg')
wrapper = HydraWrapper(
    config_path='path/to/mmseg/pspnet_config.py',
    work_dir='./work_dirs/mmseg_training'
)

config = wrapper.load_config()
results = wrapper.train(config)

# Export to ONNX
from mmdeploy.apis import torch2onnx

torch2onnx(
    img='demo/demo.jpg',
    work_dir='./work_dirs/mmseg_onnx',
    save_file='segmentation.onnx',
    deploy_cfg='configs/mmdeploy/segmentation_onnxruntime_dynamic.py',
    model_cfg=config,
    model_checkpoint='./work_dirs/mmseg_training/latest.pth',
    device='cuda:0'
)
```

## Performance Optimization Tips

### Multi-GPU Training
- Use mixed precision (AMP) to speed up training: `amp: true` in config
- Increase batch size to maximize GPU utilization
- Use gradient accumulation if batch size is limited by memory
- Pin memory for faster data loading: `pin_memory: true`

### Model Export
- Use FP16 precision for TensorRT: faster inference with minimal accuracy loss
- Batch inference when possible: export with dynamic batch dimension
- Use static shapes when input size is known: better optimization
- Profile exported model to identify bottlenecks

## Troubleshooting

### Multi-GPU Issues

**Issue**: Training hangs at initialization
- **Solution**: Check `MASTER_ADDR` and `MASTER_PORT` environment variables
- Ensure all GPUs are visible: `CUDA_VISIBLE_DEVICES=0,1,2,3`

**Issue**: Out of memory
- **Solution**: Reduce `batch_size` per GPU or use gradient accumulation
- Enable memory-efficient features in config

### Export Issues

**Issue**: ONNX export fails
- **Solution**: Check if all operators are supported by ONNX
- Use `--verify` flag to identify problematic operations
- Consider using dynamic axes for variable inputs

**Issue**: TensorRT engine creation fails
- **Solution**: Verify CUDA and TensorRT versions compatibility
- Check input shape specifications in deploy config
- Use verbose mode: `--log-level DEBUG`

## Additional Resources

- MMDeploy Documentation: https://mmdeploy.readthedocs.io/
- PyTorch Distributed Training: https://pytorch.org/tutorials/beginner/dist_overview.html
- ONNX Runtime: https://onnxruntime.ai/
- TensorRT Documentation: https://docs.nvidia.com/deeplearning/tensorrt/

## Summary

This guide covered:
- ✅ Multi-GPU training with PyTorch DistributedDataParallel
- ✅ Ray Tune multi-GPU allocation
- ✅ Model export to ONNX and TensorRT
- ✅ Integration with MMDeploy
- ✅ Production deployment workflows
- ✅ Performance optimization tips

The wrapper system provides a complete solution from training to deployment!
