# Copyright (c) OpenMMLab. All rights reserved.
"""Base wrapper class for MM libraries integration."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from mmengine.config import Config


class BaseWrapper(ABC):
    """Base wrapper class for integrating MM libraries with different frameworks.
    
    This class provides a common interface for wrapping MM libraries with
    different configuration and training frameworks like Hydra and Ray.
    
    Args:
        config_path (str): Path to the configuration file.
        work_dir (Optional[str]): Directory to save logs and models.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(
        self,
        config_path: str,
        work_dir: Optional[str] = None,
        **kwargs
    ):
        self.config_path = config_path
        self.work_dir = work_dir
        self.kwargs = kwargs
        self._mm_config = None
        
    @abstractmethod
    def load_config(self) -> Config:
        """Load and parse configuration file.
        
        Returns:
            Config: Loaded configuration object.
        """
        pass
    
    @abstractmethod
    def merge_config(self, overrides: Dict[str, Any]) -> Config:
        """Merge configuration with overrides.
        
        Args:
            overrides (Dict[str, Any]): Configuration overrides.
            
        Returns:
            Config: Merged configuration object.
        """
        pass
    
    @abstractmethod
    def setup_logging(self, config: Config) -> None:
        """Setup logging backends (e.g., wandb, tensorboard).
        
        Args:
            config (Config): Configuration object.
        """
        pass
    
    @abstractmethod
    def train(self, config: Optional[Config] = None) -> Dict[str, Any]:
        """Execute training with the given configuration.
        
        Args:
            config (Optional[Config]): Configuration object. If None, uses
                the loaded config.
                
        Returns:
            Dict[str, Any]: Training results and metrics.
        """
        pass
    
    def get_mm_config(self) -> Config:
        """Get the underlying MM config object.
        
        Returns:
            Config: MM config object.
        """
        if self._mm_config is None:
            self._mm_config = self.load_config()
        return self._mm_config
    
    def convert_to_mm_config(self, config: Dict[str, Any]) -> Config:
        """Convert a dictionary configuration to MM Config format.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary.
            
        Returns:
            Config: MM Config object.
        """
        # Load base config if it exists
        if self._mm_config is not None:
            mm_cfg = self._mm_config.copy()
        else:
            mm_cfg = Config()
        
        # Merge with new config
        mm_cfg.merge_from_dict(config)
        return mm_cfg
    
    def configure_metrics(
        self,
        config: Config,
        metric_type: str = 'bbox',
        classwise: bool = True,
        iou_thrs: Optional[list] = None,
        metric_items: Optional[list] = None
    ) -> Config:
        """Configure evaluation metrics for the config.
        
        This is a convenience method to easily override evaluation metrics.
        
        Args:
            config (Config): Configuration object to modify.
            metric_type (str): Type of metric ('bbox', 'segm', or list of both).
            classwise (bool): Whether to compute per-class metrics.
            iou_thrs (Optional[list]): List of IoU thresholds. If None, uses default.
            metric_items (Optional[list]): List of metric items to report.
            
        Returns:
            Config: Configuration with updated metrics.
        """
        metric_config = {
            'type': 'CocoMetric',
            'metric': metric_type,
            'classwise': classwise
        }
        
        if iou_thrs is not None:
            metric_config['iou_thrs'] = iou_thrs
        
        if metric_items is not None:
            metric_config['metric_items'] = metric_items
        
        # Update both val and test evaluators
        config.val_evaluator = metric_config.copy()
        config.test_evaluator = metric_config.copy()
        
        return config
