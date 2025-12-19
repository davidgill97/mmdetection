#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Export MMDetection models to ONNX or TensorRT formats.

This tool provides a simplified interface for exporting MMDetection models
to different deployment backends (ONNX, TensorRT, etc.) using MMDeploy.

Prerequisites:
    - Install MMDeploy: pip install mmdeploy
    - For TensorRT: Install TensorRT and CUDA

Example:
    Export to ONNX:
        python tools/deployment/export_model.py \\
            configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \\
            checkpoints/faster_rcnn_r50_fpn_1x_coco.pth \\
            --format onnx \\
            --work-dir work_dirs/export/faster_rcnn_onnx

    Export to TensorRT:
        python tools/deployment/export_model.py \\
            configs/retinanet/retinanet_r50_fpn_1x_coco.py \\
            checkpoints/retinanet_r50_fpn_1x_coco.pth \\
            --format tensorrt \\
            --work-dir work_dirs/export/retinanet_tensorrt \\
            --device cuda:0

    Export multiple models:
        python tools/deployment/export_model.py \\
            configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \\
            configs/retinanet/retinanet_r50_fpn_1x_coco.py \\
            --checkpoints \\
            checkpoints/faster_rcnn_r50_fpn_1x_coco.pth \\
            checkpoints/retinanet_r50_fpn_1x_coco.pth \\
            --format onnx \\
            --work-dir work_dirs/export
"""

import argparse
import os
import os.path as osp
import sys
import warnings
from pathlib import Path

import mmengine
from mmengine.logging import print_log


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export MMDetection models to ONNX/TensorRT')
    parser.add_argument(
        'config',
        type=str,
        nargs='+',
        help='Path to model config file(s). '
        'Multiple configs can be provided for batch export.')
    parser.add_argument(
        '--checkpoints',
        type=str,
        nargs='+',
        default=None,
        help='Path to model checkpoint file(s). '
        'Must match the number of config files. '
        'If not provided, will look for checkpoint in config.')
    parser.add_argument(
        '--format',
        type=str,
        default='onnx',
        choices=['onnx', 'tensorrt', 'onnxruntime'],
        help='Export format: onnx, tensorrt, or onnxruntime (default: onnx)')
    parser.add_argument(
        '--work-dir',
        type=str,
        default='work_dirs/export',
        help='Directory to save exported models (default: work_dirs/export)')
    parser.add_argument(
        '--img',
        type=str,
        default='demo/demo.jpg',
        help='Test image for model conversion (default: demo/demo.jpg)')
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device for model conversion (default: cpu). '
        'Use "cuda:0" for TensorRT export.')
    parser.add_argument(
        '--shape',
        type=str,
        default='dynamic',
        choices=['static', 'dynamic'],
        help='Input shape mode: static or dynamic (default: dynamic)')
    parser.add_argument(
        '--task-type',
        type=str,
        default='detection',
        choices=['detection', 'instance-seg'],
        help='Task type: detection or instance-seg (default: detection)')
    parser.add_argument(
        '--precision',
        type=str,
        default='fp32',
        choices=['fp32', 'fp16', 'int8'],
        help='Model precision: fp32, fp16, or int8 (default: fp32)')
    parser.add_argument(
        '--mmdeploy-dir',
        type=str,
        default=None,
        help='Path to MMDeploy repository. '
        'If not provided, assumes mmdeploy is installed and uses builtin configs.')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.checkpoints:
        if len(args.config) != len(args.checkpoints):
            raise ValueError(
                f'Number of configs ({len(args.config)}) must match '
                f'number of checkpoints ({len(args.checkpoints)})')
    
    # TensorRT requires CUDA device
    if args.format == 'tensorrt' and not args.device.startswith('cuda'):
        warnings.warn(
            'TensorRT export typically requires CUDA device. '
            f'You specified device={args.device}. '
            'This may fail. Consider using --device cuda:0')
    
    return args


def get_deploy_config(format_type, task_type, precision, shape, mmdeploy_dir):
    """Get the deployment config file path based on parameters.
    
    Args:
        format_type: Export format (onnx, tensorrt, etc.)
        task_type: Task type (detection, instance-seg)
        precision: Model precision (fp32, fp16, int8)
        shape: Shape mode (static, dynamic)
        mmdeploy_dir: Path to mmdeploy directory
    
    Returns:
        Path to deployment config file
    """
    # Map format to backend name
    backend_map = {
        'onnx': 'onnxruntime',
        'onnxruntime': 'onnxruntime',
        'tensorrt': 'tensorrt'
    }
    backend = backend_map.get(format_type, format_type)
    
    # Build config file name
    precision_str = '' if precision == 'fp32' else f'-{precision}'
    shape_str = f'_{shape}'
    
    # Default shape ranges for common configs
    if shape == 'dynamic':
        shape_range = '_320x320-1344x1344'
    else:
        shape_range = '_800x1344'
    
    config_name = f'{task_type}_{backend}{precision_str}{shape_str}{shape_range}.py'
    
    # Find config file
    if mmdeploy_dir:
        config_path = osp.join(mmdeploy_dir, 'configs', 'mmdet', 
                              task_type, config_name)
    else:
        # Try to import mmdeploy and use its config
        try:
            import mmdeploy
            mmdeploy_root = osp.dirname(osp.dirname(mmdeploy.__file__))
            config_path = osp.join(mmdeploy_root, 'configs', 'mmdet',
                                  task_type, config_name)
        except ImportError:
            raise ImportError(
                'MMDeploy is not installed. Please install it with: '
                'pip install mmdeploy\n'
                'Or specify --mmdeploy-dir to point to MMDeploy repository.')
    
    if not osp.exists(config_path):
        # Try without shape range suffix
        config_name_simple = f'{task_type}_{backend}{precision_str}{shape_str}.py'
        if mmdeploy_dir:
            config_path = osp.join(mmdeploy_dir, 'configs', 'mmdet',
                                  task_type, config_name_simple)
        else:
            import mmdeploy
            mmdeploy_root = osp.dirname(osp.dirname(mmdeploy.__file__))
            config_path = osp.join(mmdeploy_root, 'configs', 'mmdet',
                                  task_type, config_name_simple)
    
    return config_path


def export_single_model(config_file, checkpoint_file, deploy_cfg, 
                       work_dir, img_path, device, verbose=False):
    """Export a single model.
    
    Args:
        config_file: Path to model config
        checkpoint_file: Path to checkpoint
        deploy_cfg: Path to deployment config
        work_dir: Output directory
        img_path: Test image path
        device: Device for conversion
        verbose: Enable verbose output
    """
    try:
        from mmdeploy.apis import torch2onnx, torch2torchscript
        from mmdeploy.backend.sdk.export_info import export2SDK
    except ImportError:
        raise ImportError(
            'MMDeploy is not installed. Please install it with:\n'
            'pip install mmdeploy\n\n'
            'For more installation options, see:\n'
            'https://mmdeploy.readthedocs.io/en/latest/get_started.html')
    
    # Create work directory
    os.makedirs(work_dir, exist_ok=True)
    
    # Get model name for output
    model_name = osp.splitext(osp.basename(config_file))[0]
    save_file = 'end2end.onnx'
    
    print_log(f'Exporting model: {model_name}', logger='current')
    print_log(f'  Config: {config_file}', logger='current')
    print_log(f'  Checkpoint: {checkpoint_file}', logger='current')
    print_log(f'  Deploy config: {deploy_cfg}', logger='current')
    print_log(f'  Output directory: {work_dir}', logger='current')
    print_log(f'  Device: {device}', logger='current')
    
    try:
        # Convert model
        print_log('Converting model...', logger='current')
        torch2onnx(
            img=img_path,
            work_dir=work_dir,
            save_file=save_file,
            deploy_cfg=deploy_cfg,
            model_cfg=config_file,
            model_checkpoint=checkpoint_file,
            device=device
        )
        
        # Export SDK info
        print_log('Exporting SDK information...', logger='current')
        export2SDK(
            deploy_cfg=deploy_cfg,
            model_cfg=config_file,
            work_dir=work_dir,
            pth=checkpoint_file,
            device=device
        )
        
        print_log(f'✓ Successfully exported model to: {work_dir}', 
                 logger='current')
        print_log(f'  Model file: {osp.join(work_dir, save_file)}',
                 logger='current')
        
        return True
        
    except Exception as e:
        print_log(f'✗ Failed to export model: {str(e)}', 
                 logger='current', level='ERROR')
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def main():
    args = parse_args()
    
    # Check if test image exists
    if not osp.exists(args.img):
        raise FileNotFoundError(
            f'Test image not found: {args.img}\n'
            'Please specify a valid image path with --img')
    
    # Get deployment config
    try:
        deploy_cfg = get_deploy_config(
            args.format, args.task_type, args.precision, 
            args.shape, args.mmdeploy_dir)
        
        if not osp.exists(deploy_cfg):
            raise FileNotFoundError(
                f'Deployment config not found: {deploy_cfg}\n'
                'Please check your MMDeploy installation or specify '
                '--mmdeploy-dir')
        
        print_log(f'Using deployment config: {deploy_cfg}', logger='current')
        
    except Exception as e:
        print_log(f'Error getting deployment config: {str(e)}',
                 logger='current', level='ERROR')
        sys.exit(1)
    
    # Process each model
    success_count = 0
    fail_count = 0
    
    for i, config_file in enumerate(args.config):
        # Get checkpoint
        if args.checkpoints:
            checkpoint_file = args.checkpoints[i]
        else:
            # Try to infer checkpoint from config
            checkpoint_file = None
            print_log('Warning: No checkpoint specified. This may fail.',
                     logger='current', level='WARNING')
        
        # Create output directory for this model
        model_name = osp.splitext(osp.basename(config_file))[0]
        model_work_dir = osp.join(args.work_dir, model_name)
        
        # Export model
        success = export_single_model(
            config_file=config_file,
            checkpoint_file=checkpoint_file,
            deploy_cfg=deploy_cfg,
            work_dir=model_work_dir,
            img_path=args.img,
            device=args.device,
            verbose=args.verbose
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
        
        print_log('', logger='current')  # Empty line for readability
    
    # Print summary
    print_log('=' * 60, logger='current')
    print_log('Export Summary:', logger='current')
    print_log(f'  Total models: {len(args.config)}', logger='current')
    print_log(f'  Successful: {success_count}', logger='current')
    print_log(f'  Failed: {fail_count}', logger='current')
    print_log(f'  Output directory: {args.work_dir}', logger='current')
    print_log('=' * 60, logger='current')
    
    if fail_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
