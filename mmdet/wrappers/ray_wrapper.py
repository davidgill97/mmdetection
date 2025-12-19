# Copyright (c) OpenMMLab. All rights reserved.
"""Ray wrapper for MM libraries with hyperparameter tuning support."""

import os
import os.path as osp
from typing import Any, Dict, Optional, Callable

from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from .base_wrapper import BaseWrapper

try:
    import ray
    from ray import tune
    from ray.tune import Tuner
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.search import ConcurrencyLimiter
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.air import RunConfig
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


class RayWrapper(BaseWrapper):
    """Ray Tune wrapper for MM libraries with hyperparameter optimization.
    
    This wrapper integrates Ray Tune for scalable hyperparameter optimization
    with MM libraries, supporting various search algorithms and schedulers.
    
    Args:
        config_path (str): Path to the MM configuration file.
        work_dir (Optional[str]): Directory to save logs and models.
        search_space (Optional[Dict]): Hyperparameter search space for Ray Tune.
        num_samples (int): Number of trials to run.
        scheduler (str): Scheduler type ('asha', 'pbt', or None).
        search_alg (str): Search algorithm ('hyperopt', 'random', or None).
        metric (str): Metric to optimize.
        mode (str): Optimization mode ('min' or 'max').
        resources_per_trial (Optional[Dict]): Resources allocated per trial.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(
        self,
        config_path: str,
        work_dir: Optional[str] = None,
        search_space: Optional[Dict] = None,
        num_samples: int = 10,
        scheduler: str = 'asha',
        search_alg: Optional[str] = 'hyperopt',
        metric: str = 'coco/bbox_mAP',
        mode: str = 'max',
        resources_per_trial: Optional[Dict] = None,
        **kwargs
    ):
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray is not installed. Please install it with: "
                "pip install ray[tune]"
            )
        
        super().__init__(config_path, work_dir, **kwargs)
        self.search_space = search_space or {}
        self.num_samples = num_samples
        self.scheduler_type = scheduler
        self.search_alg_type = search_alg
        self.metric = metric
        self.mode = mode
        self.resources_per_trial = resources_per_trial or {"cpu": 1, "gpu": 0}
        
    def load_config(self) -> Config:
        """Load configuration from file.
        
        Returns:
            Config: Loaded configuration object.
        """
        cfg = Config.fromfile(self.config_path)
        self._mm_config = cfg
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
    
    def _create_scheduler(self):
        """Create Ray Tune scheduler based on configuration.
        
        Returns:
            Scheduler object or None.
        """
        if self.scheduler_type == 'asha':
            return ASHAScheduler(
                metric=self.metric,
                mode=self.mode,
                max_t=100,
                grace_period=10,
                reduction_factor=3
            )
        elif self.scheduler_type == 'pbt':
            return PopulationBasedTraining(
                metric=self.metric,
                mode=self.mode,
                perturbation_interval=5,
                hyperparam_mutations=self.search_space
            )
        return None
    
    def _create_search_algorithm(self):
        """Create Ray Tune search algorithm.
        
        Returns:
            Search algorithm object or None.
        """
        if self.search_alg_type == 'hyperopt':
            search_alg = HyperOptSearch(
                metric=self.metric,
                mode=self.mode
            )
            # Limit concurrent trials
            search_alg = ConcurrencyLimiter(search_alg, max_concurrent=4)
            return search_alg
        return None
    
    def _train_function(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Training function to be executed by Ray Tune.
        
        Args:
            config_updates (Dict[str, Any]): Hyperparameter configuration from Ray.
            
        Returns:
            Dict[str, Any]: Training metrics.
        """
        # Load base config
        cfg = self.get_mm_config().copy()
        
        # Merge with hyperparameter updates
        cfg.merge_from_dict(config_updates)
        
        # Setup work directory for this trial
        trial_dir = tune.get_trial_dir() if tune.is_session_enabled() else None
        if trial_dir:
            cfg.work_dir = trial_dir
        elif self.work_dir:
            cfg.work_dir = self.work_dir
        
        # Setup logging
        self.setup_logging(cfg)
        
        # Build and train the runner
        if 'runner_type' not in cfg:
            runner = Runner.from_cfg(cfg)
        else:
            runner = RUNNERS.build(cfg)
        
        # Start training
        runner.train()
        
        # Extract metrics
        metrics = {}
        if hasattr(runner, 'train_loop') and hasattr(runner.train_loop, 'evaluator'):
            metrics = runner.train_loop.evaluator.metrics
        
        # Report metrics to Ray Tune
        if tune.is_session_enabled():
            tune.report(**metrics)
        
        return metrics
    
    def train(self, config: Optional[Config] = None) -> Dict[str, Any]:
        """Execute training with Ray Tune hyperparameter optimization.
        
        Args:
            config (Optional[Config]): Configuration object. If None, uses
                the loaded config.
                
        Returns:
            Dict[str, Any]: Best trial results and metrics.
        """
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Create scheduler and search algorithm
        scheduler = self._create_scheduler()
        search_alg = self._create_search_algorithm()
        
        # Setup run config
        run_config = RunConfig(
            name="mm_ray_tune",
            local_dir=self.work_dir or "./ray_results",
        )
        
        # Create tuner
        tuner = Tuner(
            tune.with_resources(
                self._train_function,
                resources=self.resources_per_trial
            ),
            tune_config=tune.TuneConfig(
                metric=self.metric,
                mode=self.mode,
                scheduler=scheduler,
                search_alg=search_alg,
                num_samples=self.num_samples,
            ),
            run_config=run_config,
            param_space=self.search_space,
        )
        
        # Run tuning
        results = tuner.fit()
        
        # Get best trial
        best_result = results.get_best_result(metric=self.metric, mode=self.mode)
        
        return {
            'best_config': best_result.config,
            'best_metrics': best_result.metrics,
            'best_checkpoint': best_result.checkpoint,
        }
    
    def train_single(self, config: Optional[Config] = None) -> Dict[str, Any]:
        """Execute single training run without hyperparameter tuning.
        
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
            config.work_dir = './work_dirs/ray_training'
        
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
