# Copyright (c) OpenMMLab. All rights reserved.
"""Utilities for configuration conversion and management."""

from typing import Any, Dict, Optional
from pathlib import Path

from mmengine.config import Config


class ConfigConverter:
    """Utility class for converting between different configuration formats.
    
    Supports conversion between MM config, Hydra config, and dictionary formats.
    """
    
    @staticmethod
    def mm_to_dict(mm_config: Config) -> Dict[str, Any]:
        """Convert MM config to dictionary.
        
        Args:
            mm_config (Config): MM Config object.
            
        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        return mm_config.to_dict()
    
    @staticmethod
    def dict_to_mm(config_dict: Dict[str, Any]) -> Config:
        """Convert dictionary to MM config.
        
        Args:
            config_dict (Dict[str, Any]): Configuration dictionary.
            
        Returns:
            Config: MM Config object.
        """
        return Config(config_dict)
    
    @staticmethod
    def hydra_to_mm(hydra_config: Any, mm_base_path: Optional[str] = None) -> Config:
        """Convert Hydra config to MM config.
        
        Args:
            hydra_config: Hydra config object (OmegaConf DictConfig).
            mm_base_path (Optional[str]): Path to base MM config file.
            
        Returns:
            Config: MM Config object.
        """
        try:
            from omegaconf import OmegaConf
        except ImportError:
            raise ImportError("OmegaConf not installed. Install with: pip install omegaconf")
        
        # Convert to dictionary
        config_dict = OmegaConf.to_container(
            hydra_config, resolve=True, throw_on_missing=True
        )
        
        # Load base MM config if provided
        if mm_base_path and 'mm_config' in config_dict:
            mm_base = config_dict['mm_config']
            if isinstance(mm_base, str):
                base_cfg = Config.fromfile(mm_base)
            else:
                base_cfg = Config(mm_base)
            
            # Remove mm_config from dict to avoid duplication
            config_dict.pop('mm_config')
        else:
            base_cfg = Config()
        
        # Merge configurations
        base_cfg.merge_from_dict(config_dict)
        return base_cfg
    
    @staticmethod
    def extract_search_space(config: Dict[str, Any], prefix: str = "tune_") -> Dict[str, Any]:
        """Extract Ray Tune search space from configuration.
        
        Parameters prefixed with `prefix` are extracted as tunable hyperparameters.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary.
            prefix (str): Prefix for tunable parameters.
            
        Returns:
            Dict[str, Any]: Search space dictionary for Ray Tune.
        """
        search_space = {}
        
        def _extract_recursive(d: Dict, parent_key: str = ""):
            for key, value in d.items():
                full_key = f"{parent_key}.{key}" if parent_key else key
                
                if isinstance(value, dict):
                    # Check if this dict defines a tune space
                    if 'type' in value and value['type'].startswith('tune.'):
                        search_space[full_key] = value
                    else:
                        _extract_recursive(value, full_key)
                elif key.startswith(prefix):
                    # Parameter marked for tuning
                    new_key = full_key.replace(prefix, "")
                    search_space[new_key] = value
        
        _extract_recursive(config)
        return search_space
    
    @staticmethod
    def create_hydra_config_template(
        mm_config_path: str,
        output_dir: str,
        config_name: str = "config"
    ) -> Path:
        """Create a Hydra config template from an MM config file.
        
        Args:
            mm_config_path (str): Path to MM config file.
            output_dir (str): Output directory for Hydra configs.
            config_name (str): Name for the Hydra config file.
            
        Returns:
            Path: Path to created Hydra config file.
        """
        import yaml
        
        # Load MM config
        mm_cfg = Config.fromfile(mm_config_path)
        
        # Create basic Hydra structure
        hydra_cfg = {
            'defaults': [
                '_self_',
            ],
            'mm_config': mm_config_path,
            'wandb': {
                'project': 'mmdetection',
                'entity': None,
                'tags': ['training'],
            },
            'work_dir': './work_dirs/${now:%Y-%m-%d}/${now:%H-%M-%S}',
        }
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Write Hydra config
        config_file = output_path / f"{config_name}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(hydra_cfg, f, default_flow_style=False)
        
        return config_file


class ConfigManager:
    """Manager class for handling multiple MM library configurations.
    
    Provides utilities for managing configs across mmdetection, mmsegmentation,
    mmclassification, and other MM libraries.
    """
    
    SUPPORTED_LIBRARIES = [
        'mmdet',
        'mmseg',
        'mmcls',
        'mmpose',
        'mmaction',
        'mmocr',
        'mmtrack',
    ]
    
    def __init__(self, library: str = 'mmdet'):
        """Initialize ConfigManager.
        
        Args:
            library (str): MM library name (e.g., 'mmdet', 'mmseg').
        """
        if library not in self.SUPPORTED_LIBRARIES:
            raise ValueError(
                f"Library {library} not supported. "
                f"Supported libraries: {self.SUPPORTED_LIBRARIES}"
            )
        self.library = library
    
    def load_config(self, config_path: str) -> Config:
        """Load configuration for the specified MM library.
        
        Args:
            config_path (str): Path to config file.
            
        Returns:
            Config: Loaded configuration.
        """
        return Config.fromfile(config_path)
    
    def get_default_scope(self) -> str:
        """Get default scope for the MM library.
        
        Returns:
            str: Default scope name.
        """
        return self.library
    
    def validate_config(self, config: Config) -> bool:
        """Validate configuration for the MM library.
        
        Args:
            config (Config): Configuration to validate.
            
        Returns:
            bool: True if valid, False otherwise.
        """
        # Check for required fields based on library
        required_fields = ['model', 'train_dataloader', 'val_dataloader']
        
        for field in required_fields:
            if not hasattr(config, field):
                print(f"Warning: Missing required field '{field}' in config")
                return False
        
        return True
