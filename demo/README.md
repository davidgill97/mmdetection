# MMDetection Demo Scripts

This directory contains demonstration scripts for running inference with MMDetection models.

## Available Demos

### Single Image Inference

Run inference on a single image:

```bash
python demo/image_demo.py \
    demo/demo.jpg \
    configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco.pth
```

### Multi-Model Inference with Fusion

**New Feature**: Run multiple models simultaneously and fuse their predictions using Weighted Box Fusion (WBF):

```bash
python demo/demo_multi_model.py demo/demo.jpg \
    ./configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    ./configs/retinanet/retinanet_r50_fpn_1x_coco.py \
    --checkpoints \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth \
    https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_caffe_fpn_1x_coco/retinanet_r50_caffe_fpn_1x_coco_20200531-f11027c5.pth \
    --weights 1 2
```

#### Multi-Model Features

- **Multiple Models**: Specify multiple config and checkpoint files
- **Weighted Fusion**: Assign different weights to each model's predictions
- **Improved Accuracy**: Combining multiple models often improves detection performance
- **Flexible Configuration**: Control fusion parameters like IoU threshold and confidence calculation

#### Multi-Model Arguments

- `inputs`: Input image file or folder path
- `config`: One or more config files
- `--checkpoints`: One or more checkpoint files (must match config files)
- `--weights`: Weights for each model (default: equal weights)
- `--fusion-iou-thr`: IoU threshold for box matching in WBF (default: 0.55)
- `--skip-box-thr`: Skip boxes with score below this threshold (default: 0.0)
- `--conf-type`: Confidence calculation method: 'avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg' (default: 'avg')
- `--out-dir`: Output directory (default: 'outputs')
- `--device`: Device to use (default: 'cuda:0')
- `--pred-score-thr`: Score threshold for displaying predictions (default: 0.3)

#### Example: Three Models with Custom Weights

```bash
python demo/demo_multi_model.py demo/demo.jpg \
    configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    configs/retinanet/retinanet_r50_fpn_1x_coco.py \
    configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py \
    --checkpoints \
    checkpoints/faster_rcnn_r50_fpn_1x_coco.pth \
    checkpoints/retinanet_r50_fpn_1x_coco.pth \
    checkpoints/cascade_rcnn_r50_fpn_1x_coco.pth \
    --weights 1 1.5 2 \
    --fusion-iou-thr 0.6 \
    --conf-type max
```

### Video Inference

Run inference on a video file:

```bash
python demo/video_demo.py \
    demo/demo.mp4 \
    configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco.pth \
    --out result.mp4
```

### Large Image Inference

For large images that need to be processed in patches:

```bash
python demo/large_image_demo.py \
    demo/large_image.jpg \
    configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco.pth
```

### Webcam Demo

Run real-time detection on webcam feed:

```bash
python demo/webcam_demo.py \
    configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco.pth
```

## Jupyter Notebooks

Interactive tutorials are available in Jupyter notebook format:

- `MMDet_Tutorial.ipynb`: Basic detection tutorial
- `MMDet_InstanceSeg_Tutorial.ipynb`: Instance segmentation tutorial
- `inference_demo.ipynb`: Inference examples

To run the notebooks:

```bash
jupyter notebook
```

## Model Export

For deploying models to production, see the [Export Guide](../tools/deployment/README.md) for information on exporting models to ONNX and TensorRT formats.

## Tips

### Batch Processing

To process multiple images at once:

```bash
python demo/image_demo.py \
    path/to/image/directory \
    configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco.pth \
    --out-dir outputs
```

### Using Remote Checkpoints

You can use URLs for checkpoints instead of local files:

```bash
python demo/image_demo.py \
    demo/demo.jpg \
    configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
```

### GPU Acceleration

To use GPU for inference:

```bash
python demo/image_demo.py \
    demo/demo.jpg \
    configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco.pth \
    --device cuda:0
```

## References

- [MMDetection Documentation](https://mmdetection.readthedocs.io/)
- [Weighted Boxes Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
