# Copyright (c) OpenMMLab. All rights reserved.
"""Hydra wrapper for MM libraries."""

import os
import os.path as osp
from typing import Any, Dict, Optional

from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from .base_wrapper import BaseWrapper

try:
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    # Create dummy types for type hints when Hydra is not installed
    DictConfig = dict


class HydraWrapper(BaseWrapper):
    """Hydra wrapper for MM libraries.
    
    This wrapper integrates Hydra configuration system with MM libraries,
    allowing for hierarchical configuration composition and easy overrides.
    
    Args:
        config_path (str): Path to the Hydra configuration directory or MM config file.
        config_name (str): Name of the Hydra config file (without .yaml extension).
            If None, assumes config_path is an MM config file.
        work_dir (Optional[str]): Directory to save logs and models.
        overrides (Optional[list]): List of Hydra overrides.
        version_base (Optional[str]): Hydra version base for compatibility.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(
        self,
        config_path: str,
        config_name: Optional[str] = None,
        work_dir: Optional[str] = None,
        overrides: Optional[list] = None,
        version_base: Optional[str] = None,
        **kwargs
    ):
        super().__init__(config_path, work_dir, **kwargs)
        self.config_name = config_name
        self.overrides = overrides or []
        self.version_base = version_base
        self._hydra_cfg = None
        self.is_hydra_config = config_name is not None
        
        # Only check for Hydra if trying to use Hydra config
        if self.is_hydra_config and not HYDRA_AVAILABLE:
            raise ImportError(
                "Hydra is not installed. Please install it with: "
                "pip install hydra-core"
            )
        
    def load_config(self) -> Config:
        """Load configuration using Hydra or MM config system.
        
        Returns:
            Config: Loaded configuration object.
        """
        if self.is_hydra_config:
            # Load using Hydra
            self._hydra_cfg = self._load_hydra_config()
            # Convert Hydra config to MM config
            cfg_dict = OmegaConf.to_container(
                self._hydra_cfg, resolve=True, throw_on_missing=True
            )
            mm_cfg = self._convert_hydra_to_mm_config(cfg_dict)
        else:
            # Load MM config directly
            mm_cfg = Config.fromfile(self.config_path)
        
        self._mm_config = mm_cfg
        return mm_cfg
    
    def _load_hydra_config(self) -> DictConfig:
        """Load configuration using Hydra.
        
        Returns:
            DictConfig: Hydra configuration object.
        """
        # Get absolute path for config directory
        if osp.isabs(self.config_path):
            config_dir = self.config_path
        else:
            config_dir = osp.abspath(self.config_path)
        
        # Initialize Hydra with the config directory
        with initialize_config_dir(
            config_dir=config_dir,
            version_base=self.version_base
        ):
            cfg = compose(config_name=self.config_name, overrides=self.overrides)
        
        return cfg
    
    def _convert_hydra_to_mm_config(self, hydra_cfg: Dict[str, Any]) -> Config:
        """Convert Hydra config to MM config format.
        
        Args:
            hydra_cfg (Dict[str, Any]): Hydra configuration dictionary.
            
        Returns:
            Config: MM Config object.
        """
        # If hydra_cfg contains an 'mm_config' field, use it as base
        if 'mm_config' in hydra_cfg:
            mm_base = hydra_cfg['mm_config']
            if isinstance(mm_base, str):
                # Load MM config file
                cfg = Config.fromfile(mm_base)
            else:
                cfg = Config(mm_base)
        else:
            cfg = Config()
        
        # Merge other Hydra config fields
        for key, value in hydra_cfg.items():
            if key != 'mm_config' and key != 'hydra':
                cfg[key] = value
        
        return cfg
    
    def merge_config(self, overrides: Dict[str, Any]) -> Config:
        """Merge configuration with overrides.
        
        Args:
            overrides (Dict[str, Any]): Configuration overrides.
            
        Returns:
            Config: Merged configuration object.
        """
        cfg = self.get_mm_config()
        cfg.merge_from_dict(overrides)
        return cfg
    
    def setup_logging(self, config: Config) -> None:
        """Setup logging backends including wandb.
        
        Args:
            config (Config): Configuration object.
        """
        # Setup vis_backends if not already configured
        if not hasattr(config, 'vis_backends') or config.vis_backends is None:
            config.vis_backends = [dict(type='LocalVisBackend')]
        
        # Add wandb backend if wandb config is provided
        if hasattr(config, 'wandb') and config.wandb is not None:
            wandb_config = config.wandb
            if isinstance(wandb_config, dict):
                wandb_backend = dict(
                    type='WandbVisBackend',
                    init_kwargs=wandb_config
                )
                # Check if wandb backend already exists
                has_wandb = any(
                    backend.get('type') == 'WandbVisBackend'
                    for backend in config.vis_backends
                )
                if not has_wandb:
                    config.vis_backends.append(wandb_backend)
        
        # Update visualizer with vis_backends
        if hasattr(config, 'visualizer'):
            config.visualizer['vis_backends'] = config.vis_backends
    
    def train(self, config: Optional[Config] = None) -> Dict[str, Any]:
        """Execute training with the given configuration.
        
        Args:
            config (Optional[Config]): Configuration object. If None, uses
                the loaded config.
                
        Returns:
            Dict[str, Any]: Training results and metrics.
        """
        if config is None:
            config = self.get_mm_config()
        
        # Setup work directory
        if self.work_dir is not None:
            config.work_dir = self.work_dir
        elif not hasattr(config, 'work_dir') or config.work_dir is None:
            config.work_dir = './work_dirs/hydra_training'
        
        # Setup logging
        self.setup_logging(config)
        
        # Build the runner from config
        if 'runner_type' not in config:
            runner = Runner.from_cfg(config)
        else:
            runner = RUNNERS.build(config)
        
        # Start training
        runner.train()
        
        # Return training metrics
        results = {}
        if hasattr(runner, 'train_loop') and hasattr(runner.train_loop, 'evaluator'):
            results = runner.train_loop.evaluator.metrics
        
        return results
    
    def get_hydra_config(self) -> Optional[DictConfig]:
        """Get the Hydra configuration object.
        
        Returns:
            Optional[DictConfig]: Hydra config object or None if not using Hydra.
        """
        return self._hydra_cfg
