"""TabPFN Prior DataLoader - Standalone Implementation

This module provides a standalone implementation of TabPFN's prior generation
capabilities, extracted from the original TabPFN codebase and adapted for
easy integration into other projects.
"""

import math
import random
from functools import partial
from typing import Dict, Union, Optional

import torch
from torch.utils.data import DataLoader

# Import TabPFN priors 
from . import priors


class TabPFNPriorDataLoader(DataLoader):
    """DataLoader sampling synthetic prior data from TabPFN's priors.
    
    This class creates a DataLoader that generates synthetic tabular data using
    TabPFN's prior generation methods (MLP, GP, or prior bag combinations).
    
    Args:
        prior_type (str): Type of prior to use ('mlp', 'gp', 'gp_mix', 'prior_bag').
        num_steps (int): Number of batches per epoch.
        batch_size (int): Number of functions per batch.
        num_datapoints_max (int): Maximum number of datapoints per function.
        num_features (int): Number of input features.
        max_num_classes (int): Maximum number of classes for classification.
        device (torch.device): Target device for tensors.
        prior_config (dict, optional): Configuration for the prior hyperparameters.
        flexible (bool): Whether to use flexible categorical encoding.
        differentiable (bool): Whether to use differentiable hyperparameters.
        return_categorical_mask (bool): Whether to return categorical feature mask.
        nan_handling (bool): Whether to enable NaN handling in differentiable hyperparameters.
        **kwargs: Additional arguments passed to the prior.
    """
    
    def __init__(
        self,
        prior_type: str,
        num_steps: int,
        batch_size: int,
        num_datapoints_max: int,
        num_features: int,
        max_num_classes: int,
        device: torch.device,
        prior_config: dict = None,
        flexible: bool = True,
        differentiable: bool = False,
        return_categorical_mask: bool = False,
        nan_handling: bool = True,
        **kwargs
    ):
        self.prior_type = prior_type
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_datapoints_max = num_datapoints_max
        self.num_features = num_features
        self.max_num_classes = max_num_classes
        self.device = device
        self.flexible = flexible
        self.differentiable = differentiable
        self.return_categorical_mask = return_categorical_mask
        self.nan_handling = nan_handling
        
        self.config = self._get_default_config()
            
        if prior_config:
            self.config.update(prior_config)
            
        # Set up prior hyperparameters and get_batch function
        self.prior_hyperparameters, self.get_batch_fn = self._setup_prior()

    def _get_default_config(self) -> dict:
        """Get default configuration for TabPFN priors."""
        return {
            # MLP Prior defaults
            'prior_mlp_hidden_dim': 64,
            'prior_mlp_activations': lambda: torch.nn.ReLU(),
            'mix_activations': False,
            'noise_std': 0.1,
            'prior_mlp_dropout_prob': 0.0,
            'init_std': 1.0,
            'prior_mlp_scale_weights_sqrt': True,
            'random_feature_rotation': True,
            'is_causal': False,
            'num_causes': 0,
            'y_is_effect': False,
            'pre_sample_causes': False,
            'pre_sample_weights': False,
            'block_wise_dropout': False,
            'add_uninformative_features': False,
            'sort_features': False,
            'in_clique': False,
            
            # GP Prior defaults  
            'noise': 0.1,
            'outputscale': 1.0,
            'lengthscale': 0.2,
            'is_binary_classification': False,
            'normalize_by_used_features': True,
            'order_y': False,
            'sampling': 'uniform',
            
            # GP Mix defaults
            'lengthscale_concentration': 1.0,
            'nu': 2.5,
            'outputscale_concentration': 1.0,
            'categorical_data': False,
            'y_minmax_norm': True,
            'noise_concentration': 1.0,
            'noise_rate': 1.0,
            
            # General defaults
            'num_layers': 2,
            'verbose': False,
            'emsize': 256,  # Embedding size for differentiable prior
            
            # Flexible categorical defaults
            'seq_len_used': None,  # Will be set to num_datapoints_max
            'num_features_used': None,  # Will be set to num_features
            'balanced': True,
            'max_num_classes': 2,
            'num_classes': 2,
            'categorical_feature_p': 0.0,
            'nan_prob_no_reason': 0.0,
            'nan_prob_unknown_reason': 0.0,
            'nan_prob_a_reason': 0.0,
            'nan_prob_unknown_reason_reason_prior': 0.0,
            'set_value_to_nan': 1.0,  # should not do anything if nan_prob_* are 0
            'normalize_to_ranking': False,
            'noise_type': "Gaussian",
            'output_multiclass_ordered_p': 0.0,
            'multiclass_type': 'rank',
            'normalize_labels': True,
            'check_is_compatible': True,
            'rotate_normalized_labels': True,
        }
    
    def _setup_prior(self):
        """Set up the prior hyperparameters and get_batch function based on prior_type."""
        
        def make_get_batch(model_proto, **extra_kwargs):
            """Create a get_batch function for the given prior."""
            def new_get_batch(batch_size, seq_len, num_features, hyperparameters, device, **kwargs):
                kwargs = {**extra_kwargs, **kwargs}
                return model_proto.get_batch(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    device=device,
                    hyperparameters=hyperparameters,
                    num_features=num_features,
                    **kwargs
                )
            return new_get_batch
        
        if self.prior_type == 'prior_bag':
            # Prior bag combines GP and MLP priors
            get_batch_gp = make_get_batch(priors.fast_gp)
            get_batch_mlp = make_get_batch(priors.mlp)
            
            if self.flexible:
                get_batch_gp = make_get_batch(priors.flexible_categorical, get_batch=get_batch_gp)
                get_batch_mlp = make_get_batch(priors.flexible_categorical, get_batch=get_batch_mlp)
            
            prior_bag_hyperparameters = {
                'prior_bag_get_batch': (get_batch_gp, get_batch_mlp),
                'prior_bag_exp_weights_1': 2.0
            }
            
            # Combine MLP and GP hyperparameters
            mlp_hyperparameters = self._get_mlp_prior_hyperparameters()
            gp_hyperparameters = self._get_gp_prior_hyperparameters()
            prior_hyperparameters = {**mlp_hyperparameters, **gp_hyperparameters, **prior_bag_hyperparameters}
            
            model_proto = priors.prior_bag
            
        elif self.prior_type == 'mlp':
            prior_hyperparameters = self._get_mlp_prior_hyperparameters()
            model_proto = priors.mlp
            
        elif self.prior_type == 'gp':
            prior_hyperparameters = self._get_gp_prior_hyperparameters()
            model_proto = priors.fast_gp
            
        elif self.prior_type == 'gp_mix':
            prior_hyperparameters = self._get_gp_mix_prior_hyperparameters()
            model_proto = priors.fast_gp_mix
            
        else:
            raise ValueError(f"Unsupported prior type: {self.prior_type}")
        
        # Apply flexible categorical wrapper if needed
        extra_kwargs = {}
        if self.flexible and self.prior_type != 'prior_bag':
            get_batch_base = make_get_batch(model_proto)
            extra_kwargs['get_batch'] = get_batch_base
            model_proto = priors.flexible_categorical
        
        # Add flexible categorical specific parameters and ensure all required params are set
        if self.flexible:
            # Get flexible categorical parameters from config
            flex_params = self._get_flexible_categorical_hyperparameters()
            prior_hyperparameters.update(flex_params)
            
            # Override with instance-specific values
            prior_hyperparameters['seq_len_used'] = self.num_datapoints_max
            prior_hyperparameters['num_features_used'] = lambda: self.num_features
            prior_hyperparameters['max_num_classes'] = self.max_num_classes
            prior_hyperparameters['num_classes'] = self.max_num_classes
            
            # Fix balanced multiclass issue - balanced training only works for binary classification
            if self.max_num_classes > 2:
                prior_hyperparameters['balanced'] = False
                
            # Special handling for activations in flexible categorical context
            # Create a factory function that always returns a fresh activation class
            if 'prior_mlp_activations' in prior_hyperparameters:
                activation_class = torch.nn.ReLU  # Store the class
                prior_hyperparameters['prior_mlp_activations'] = lambda: activation_class
        
        # Apply differentiable wrapper if needed
        if self.differentiable:
            get_batch_base = make_get_batch(model_proto, **extra_kwargs)
            extra_kwargs = {'get_batch': get_batch_base, 'differentiable_hyperparameters': self._get_differentiable_hyperparameters()}
            model_proto = priors.differentiable_prior
        
        # Create final get_batch function
        get_batch_fn = make_get_batch(model_proto, **extra_kwargs)
        
        return prior_hyperparameters, get_batch_fn
    
    def _get_mlp_prior_hyperparameters(self) -> dict:
        """Get MLP prior hyperparameters from config."""
        mlp_params = {}
        mlp_keys = [
            'prior_mlp_hidden_dim', 'prior_mlp_activations', 'mix_activations', 'noise_std', 
            'prior_mlp_dropout_prob', 'init_std', 'prior_mlp_scale_weights_sqrt',
            'random_feature_rotation', 'is_causal', 'num_causes', 'y_is_effect',
            'pre_sample_causes', 'pre_sample_weights', 'block_wise_dropout',
            'add_uninformative_features', 'sort_features', 'in_clique', 'num_layers',
            'sampling', 'emsize'
        ]
        
        for key in mlp_keys:
            if key in self.config:
                mlp_params[key] = self.config[key]
                
        return mlp_params
    
    def _get_gp_prior_hyperparameters(self) -> dict:
        """Get GP prior hyperparameters from config."""
        gp_params = {}
        gp_keys = [
            'noise', 'outputscale', 'lengthscale', 'is_binary_classification',
            'normalize_by_used_features', 'order_y', 'sampling', 'emsize'
        ]
        
        for key in gp_keys:
            if key in self.config:
                gp_params[key] = self.config[key]
                
        return gp_params
    
    def _get_gp_mix_prior_hyperparameters(self) -> dict:
        """Get GP Mix prior hyperparameters from config."""
        gp_mix_params = {}
        gp_mix_keys = [
            'lengthscale_concentration', 'nu', 'outputscale_concentration',
            'categorical_data', 'y_minmax_norm', 'noise_concentration', 'noise_rate'
        ]
        
        for key in gp_mix_keys:
            if key in self.config:
                gp_mix_params[key] = self.config[key]
                
        return gp_mix_params
    
    def _get_flexible_categorical_hyperparameters(self) -> dict:
        """Get flexible categorical hyperparameters from config."""
        flex_params = {}
        flex_keys = [
            'seq_len_used', 'num_features_used', 'balanced', 'max_num_classes', 'num_classes',
            'categorical_feature_p', 'nan_prob_no_reason', 'nan_prob_unknown_reason', 
            'nan_prob_a_reason', 'nan_prob_unknown_reason_reason_prior', 'set_value_to_nan', 
            'missing_values', 'normalize_to_ranking', 'noise_type',
            'output_multiclass_ordered_p', 'multiclass_type', 'normalize_labels',
            'check_is_compatible', 'rotate_normalized_labels', 'normalize_by_used_features',
            'emsize'
        ]
        
        for key in flex_keys:
            if key in self.config:
                flex_params[key] = self.config[key]
                
        return flex_params
    
    def _get_differentiable_hyperparameters(self) -> dict:
        """Get differentiable hyperparameters for advanced optimization."""
        # Based on TabPFN's model_configs.py differentiable hyperparameters
        diff_hyperparameters = {}
        
        # MLP/Causal differentiable parameters
        if self.prior_type in ['mlp', 'prior_bag']:
            diff_hyperparameters.update({
                'num_layers': {'distribution': 'meta_gamma', 'max_alpha': 2, 'max_scale': 3, 'round': True, 'lower_bound': 2},
                'prior_mlp_hidden_dim': {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 100, 'round': True, 'lower_bound': 4},
                'prior_mlp_dropout_prob': {'distribution': 'meta_beta', 'scale': 0.6, 'min': 0.1, 'max': 5.0},
                'noise_std': {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 0.3, 'min_mean': 0.0001, 'round': False, 'lower_bound': 0.0},
                'init_std': {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 10.0, 'min_mean': 0.01, 'round': False, 'lower_bound': 0.0},
                'num_causes': {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 7, 'round': True, 'lower_bound': 2},
                'is_causal': {'distribution': 'meta_choice', 'choice_values': [True, False]},
                'pre_sample_weights': {'distribution': 'meta_choice', 'choice_values': [True, False]},
                'y_is_effect': {'distribution': 'meta_choice', 'choice_values': [True, False]},
                'sampling': {'distribution': 'meta_choice', 'choice_values': ['normal', 'mixed']},
                'prior_mlp_activations': {'distribution': 'meta_choice_mixed', 'choice_values': [torch.nn.Tanh, torch.nn.Identity, torch.nn.ReLU]},
                'block_wise_dropout': {'distribution': 'meta_choice', 'choice_values': [True, False]},
                'sort_features': {'distribution': 'meta_choice', 'choice_values': [True, False]},
                'in_clique': {'distribution': 'meta_choice', 'choice_values': [True, False]},
            })
        
        # GP differentiable parameters
        if self.prior_type in ['gp', 'gp_mix', 'prior_bag']:
            diff_hyperparameters.update({
                'outputscale': {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 10.0, 'min_mean': 0.00001, 'round': False, 'lower_bound': 0},
                'lengthscale': {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 10.0, 'min_mean': 0.00001, 'round': False, 'lower_bound': 0},
                'noise': {'distribution': 'meta_choice', 'choice_values': [0.00001, 0.0001, 0.01]}
            })
        
        # Flexible categorical differentiable parameters
        if self.flexible:
            diff_hyperparameters.update({
                'output_multiclass_ordered_p': {'distribution': 'uniform', 'min': 0.0, 'max': 0.5},
                'multiclass_type': {'distribution': 'meta_choice', 'choice_values': ['value', 'rank']},
                'categorical_feature_p': {'distribution': 'meta_beta', 'scale': 1.0, 'min': 0.5, 'max': 3.0},
                'normalize_to_ranking': {'distribution': 'meta_choice', 'choice_values': [True, False]},
            })
            
        if self.nan_handling:
            diff_hyperparameters.update({
                'nan_prob_no_reason': {'distribution': 'uniform', 'min': 0.0, 'max': 0.1},
                #'nan_prob_a_reason': {'distribution': 'uniform', 'min': 0.0, 'max': 0.1},
                #'nan_prob_unknown_reason': {'distribution': 'uniform', 'min': 0.0, 'max': 0.1},
                #'nan_prob_unknown_reason_reason_prior': {'distribution': 'uniform', 'min': 0.0, 'max': 1.0},
                # set_value_to_nan to be a constant 1.0
                #'set_value_to_nan': {'distribution': 'meta_choice', 'choice_values': [1.0, 1.0]},
            })
    
        # Prior bag specific parameters
        if self.prior_type == 'prior_bag':
            diff_hyperparameters.update({
                'prior_bag_exp_weights_1': {'distribution': 'uniform', 'min': 2.0, 'max': 10.0},
            })
        
        return diff_hyperparameters
    
    def _tabpfn_to_ours(self, batch_data, single_eval_pos) -> Dict[str, Union[torch.Tensor, int]]:
        """Convert TabPFN batch format to our standard format."""
        if len(batch_data) == 3:
            # flexible_categorical returns (x, y, y_)
            x, y, target_y = batch_data
        elif len(batch_data) == 4:
            # Some priors might return (x, y, target_y, single_eval_pos)
            x, y, target_y, single_eval_pos = batch_data
        else:
            raise ValueError(f"Unexpected batch_data format with {len(batch_data)} elements")
        
        # TabPFN priors return data in (seq_len, batch_size, num_features) format
        # but we need (batch_size, seq_len, num_features) format
        if len(x.shape) == 3:
            x = x.transpose(0, 1)  # (seq_len, batch_size, num_features) -> (batch_size, seq_len, num_features)
        if len(y.shape) == 2:
            y = y.transpose(0, 1)  # (seq_len, batch_size) -> (batch_size, seq_len)
        if len(target_y.shape) == 2:
            target_y = target_y.transpose(0, 1)  # (seq_len, batch_size) -> (batch_size, seq_len)
        
        result = {
            'x': x.to(self.device),
            'y': y.to(self.device), 
            'target_y': target_y.to(self.device),
            'single_eval_pos': single_eval_pos
        }
        
        # Add categorical mask if available (from BatchResult)
        if self.return_categorical_mask and hasattr(batch_data, 'categorical_mask'):
            result['categorical_mask'] = batch_data.categorical_mask.to(self.device)
        
        return result
    
    def __iter__(self):
        """Generate batches of synthetic data."""
        for _ in range(self.num_steps):
            # Generate a random evaluation position (typically around 80% of sequence length)
            single_eval_pos = random.randint(
                max(1, int(0.1 * self.num_datapoints_max)), 
                int(0.9 * self.num_datapoints_max)
            )
            
            # Generate batch using TabPFN's get_batch method
            batch_data = self.get_batch_fn(
                batch_size=self.batch_size,
                seq_len=self.num_datapoints_max,
                num_features=self.num_features,
                hyperparameters=self.prior_hyperparameters,
                device=self.device,
                num_outputs=1,
                single_eval_pos=single_eval_pos
            )
            
            yield self._tabpfn_to_ours(batch_data, single_eval_pos)
    
    def __len__(self) -> int:
        return self.num_steps


def build_tabpfn_prior(
    prior_type: str = 'mlp',
    num_steps: int = 100,
    batch_size: int = 8,
    num_datapoints_max: int = 1024,
    num_features: int = 10,
    max_num_classes: int = 10,
    device: str = 'cpu',
    prior_config: dict = None,
    flexible: bool = True,
    differentiable: bool = False,
    return_categorical_mask: bool = False,
    nan_handling: bool = False,
    **kwargs
) -> TabPFNPriorDataLoader:
    """Build a TabPFN prior dataloader with the specified configuration.
    
    Args:
        prior_type (str): Type of prior ('mlp', 'gp', 'gp_mix', 'prior_bag').
        num_steps (int): Number of batches per epoch.
        batch_size (int): Number of functions per batch.
        num_datapoints_max (int): Maximum sequence length per function.
        num_features (int): Number of input features.
        max_num_classes (int): Maximum number of classes.
        device (str): Device to use ('cpu' or 'cuda').
        prior_config (dict): Custom prior configuration.
        flexible (bool): Whether to use flexible categorical encoding.
        differentiable (bool): Whether to use differentiable hyperparameters.
        return_categorical_mask (bool): If True and flexible=True, batches include
            'categorical_mask' key with boolean tensor [num_features] indicating
            which features are categorical (default: False).
        nan_handling (bool): Whether to enable NaN handling in differentiable hyperparameters.
        **kwargs: Additional arguments.
        
    Returns:
        TabPFNPriorDataLoader: Configured TabPFN prior dataloader.
    """
    device = torch.device(device)
    
    return TabPFNPriorDataLoader(
        prior_type=prior_type,
        num_steps=num_steps,
        batch_size=batch_size,
        num_datapoints_max=num_datapoints_max,
        num_features=num_features,
        max_num_classes=max_num_classes,
        device=device,
        prior_config=prior_config,
        flexible=flexible,
        differentiable=differentiable,
        return_categorical_mask=return_categorical_mask,
        nan_handling=nan_handling,
        **kwargs
    )