# Model Export Guide

This directory contains tools for exporting MMDetection models to various deployment formats.

## Quick Start

### Export to ONNX

The simplest way to export a model to ONNX format:

```bash
python tools/deployment/export_model.py \
    configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco.pth \
    --format onnx \
    --work-dir work_dirs/export/faster_rcnn_onnx
```

### Export to TensorRT

To export to TensorRT format (requires CUDA):

```bash
python tools/deployment/export_model.py \
    configs/retinanet/retinanet_r50_fpn_1x_coco.py \
    checkpoints/retinanet_r50_fpn_1x_coco.pth \
    --format tensorrt \
    --work-dir work_dirs/export/retinanet_tensorrt \
    --device cuda:0
```

### Export Multiple Models

You can export multiple models in a single command:

```bash
python tools/deployment/export_model.py \
    configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    configs/retinanet/retinanet_r50_fpn_1x_coco.py \
    --checkpoints \
    checkpoints/faster_rcnn_r50_fpn_1x_coco.pth \
    checkpoints/retinanet_r50_fpn_1x_coco.pth \
    --format onnx \
    --work-dir work_dirs/export
```

## Prerequisites

### Install MMDeploy

```bash
pip install mmdeploy
```

For more installation options and backend support, see the [MMDeploy Installation Guide](https://mmdeploy.readthedocs.io/en/latest/get_started.html).

### Backend-Specific Requirements

- **ONNX**: No additional requirements (included with mmdeploy)
- **TensorRT**: Requires NVIDIA GPU, CUDA, and TensorRT installation
- **Other backends**: See [MMDeploy documentation](https://mmdeploy.readthedocs.io/en/latest/)

## Usage

### Command-Line Arguments

- `config`: Path to model config file(s). Multiple configs can be provided for batch export.
- `--checkpoints`: Path to model checkpoint file(s). Must match the number of config files.
- `--format`: Export format: `onnx`, `tensorrt`, or `onnxruntime` (default: `onnx`)
- `--work-dir`: Directory to save exported models (default: `work_dirs/export`)
- `--img`: Test image for model conversion (default: `demo/demo.jpg`)
- `--device`: Device for model conversion (default: `cpu`). Use `cuda:0` for TensorRT.
- `--shape`: Input shape mode: `static` or `dynamic` (default: `dynamic`)
- `--task-type`: Task type: `detection` or `instance-seg` (default: `detection`)
- `--precision`: Model precision: `fp32`, `fp16`, or `int8` (default: `fp32`)
- `--mmdeploy-dir`: Path to MMDeploy repository (optional)
- `--verbose`: Enable verbose output

### Examples

#### Export Detection Model

```bash
python tools/deployment/export_model.py \
    configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco.pth \
    --format onnx
```

#### Export Instance Segmentation Model

```bash
python tools/deployment/export_model.py \
    configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py \
    checkpoints/mask_rcnn_r50_fpn_1x_coco.pth \
    --format onnx \
    --task-type instance-seg
```

#### Export with FP16 Precision

```bash
python tools/deployment/export_model.py \
    configs/retinanet/retinanet_r50_fpn_1x_coco.py \
    checkpoints/retinanet_r50_fpn_1x_coco.pth \
    --format tensorrt \
    --precision fp16 \
    --device cuda:0
```

#### Export with Custom Test Image

```bash
python tools/deployment/export_model.py \
    configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco.pth \
    --format onnx \
    --img path/to/your/test/image.jpg
```

## Output Structure

After successful export, the output directory will contain:

```
work_dirs/export/model_name/
├── deploy.json          # Deployment configuration
├── detail.json          # Model details
├── end2end.onnx        # Exported model file
└── pipeline.json        # Inference pipeline info
```

## Running Multiple Models

The `demo/demo_multi_model.py` script allows you to run inference with multiple models simultaneously and fuse their results:

```bash
python demo/demo_multi_model.py demo/demo.jpg \
    ./configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    ./configs/retinanet/retinanet_r50_fpn_1x_coco.py \
    --checkpoints \
    checkpoints/faster_rcnn_r50_fpn_1x_coco.pth \
    checkpoints/retinanet_r50_fpn_1x_coco.pth \
    --weights 1 2
```

This uses Weighted Box Fusion (WBF) to combine predictions from multiple models.

## Troubleshooting

### MMDeploy Not Found

If you get an error about MMDeploy not being installed:

```bash
pip install mmdeploy
```

If you built MMDeploy from source, specify the path:

```bash
python tools/deployment/export_model.py \
    config.py checkpoint.pth \
    --mmdeploy-dir /path/to/mmdeploy
```

### TensorRT Export Fails

Make sure you're using a CUDA device:

```bash
python tools/deployment/export_model.py \
    config.py checkpoint.pth \
    --format tensorrt \
    --device cuda:0  # Important!
```

### Deployment Config Not Found

The tool automatically finds the appropriate deployment config based on your settings. If it fails, you may need to:

1. Install MMDeploy properly: `pip install mmdeploy`
2. Or clone the MMDeploy repository and specify `--mmdeploy-dir`

## Advanced Usage

### Custom Deployment Config

If you need more control, you can use the MMDeploy APIs directly:

```python
from mmdeploy.apis import torch2onnx
from mmdeploy.backend.sdk.export_info import export2SDK

img = 'demo/demo.jpg'
work_dir = 'work_dirs/export/custom'
save_file = 'end2end.onnx'
deploy_cfg = 'path/to/custom/deploy_config.py'
model_cfg = 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
model_checkpoint = 'checkpoints/faster_rcnn_r50_fpn_1x_coco.pth'
device = 'cpu'

# Convert model to ONNX
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg,
           model_checkpoint, device)

# Extract pipeline info for SDK
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint,
           device=device)
```

## References

- [MMDeploy Documentation](https://mmdeploy.readthedocs.io/)
- [MMDeploy MMDetection Support](https://mmdeploy.readthedocs.io/en/latest/04-supported-codebases/mmdet.html)
- [ONNX](https://onnx.ai/)
- [TensorRT](https://developer.nvidia.com/tensorrt)
