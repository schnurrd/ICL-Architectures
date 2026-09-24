# TabPFN v1 Prior - Standalone Prior Generation Package

A standalone, self-contained implementation of TabPFN's prior generation capabilities, extracted from the original TabPFN codebase and packaged for easy integration into other projects.

## Features

- **Complete TabPFN Prior Suite**: All major prior types from TabPFN v1
  - MLP Prior: Multi-layer perceptron based synthetic data generation
  - GP Prior: Gaussian Process based synthetic data generation  
  - GP Mix Prior: Mixture of Gaussian Processes
  - Prior Bag: Intelligent combination of multiple priors
- **Flexible Categorical Encoding**: Advanced categorical data handling and preprocessing
- **Differentiable Hyperparameters**: Research-grade hyperparameter optimization support
- **Self-Contained**: No dependencies on the full TabPFN codebase
- **Easy Integration**: Simple import and use in any Python project
- **Consistent Interface**: Compatible with tabularpriors and other data generation frameworks

## Installation

Simply copy the `tabpfn-v1-prior` folder to your project or add it to your Python path.

### Dependencies

```bash
pip install torch numpy scipy gpytorch
```

## Quick Start

```python
from tabpfn_v1_prior import build_tabpfn_prior

# Create a simple MLP prior
prior = build_tabpfn_prior(
    prior_type='mlp',
    num_steps=100,
    batch_size=8,
    num_datapoints_max=1024,
    num_features=10,
    max_num_classes=5,
    device='cpu'
)

# Generate synthetic data
for batch in prior:
    x = batch['x']          # Input features [batch_size, seq_len, num_features]
    y = batch['y']          # Labels [batch_size, seq_len]
    target_y = batch['target_y']  # Target labels (same as y)
    eval_pos = batch['single_eval_pos']  # Evaluation position
    break

print(f"Generated data: x.shape={x.shape}, y.shape={y.shape}")
```

## Prior Types

### MLP Prior (`'mlp'`)
Generates synthetic tabular data using multi-layer perceptron models with configurable:
- Hidden dimensions and layer counts
- Activation functions (Tanh, ReLU, Identity)
- Noise levels and dropout
- Causal relationships between features

```python
prior = build_tabpfn_prior(
    prior_type='mlp',
    prior_config={
        'prior_mlp_hidden_dim': 128,
        'num_layers': 3,
        'noise_std': 0.05,
        'is_causal': True
    }
)
```

### GP Prior (`'gp'`)
Generates data using Gaussian Process models with configurable:
- Lengthscale and output scale parameters
- Noise levels
- Kernel parameters

```python
prior = build_tabpfn_prior(
    prior_type='gp',
    prior_config={
        'lengthscale': 0.1,
        'outputscale': 2.0,
        'noise': 0.01
    }
)
```

### Prior Bag (`'prior_bag'`)
Intelligently combines multiple priors (MLP + GP) with automatic weighting:

```python
prior = build_tabpfn_prior(
    prior_type='prior_bag',
    flexible=True,
    differentiable=True
)
```

### Differentiable Priors
Enable advanced hyperparameter optimization with learnable distributions:

```python
prior = build_tabpfn_prior(
    prior_type='mlp',
    differentiable=True,  # Enable differentiable hyperparameters
    flexible=True
)
```

## Advanced Configuration

### Custom Hyperparameters

```python
custom_config = {
    # MLP settings
    'prior_mlp_hidden_dim': 256,
    'prior_mlp_activations': lambda: torch.nn.ReLU(),
    'num_layers': 4,
    'noise_std': 0.02,
    
    # GP settings  
    'lengthscale': 0.05,
    'outputscale': 3.0,
    
    # Flexible categorical settings
    'categorical_feature_p': 0.2,
    'normalize_by_used_features': True,
    'balanced': False
}

prior = build_tabpfn_prior(
    prior_type='prior_bag',
    prior_config=custom_config,
    flexible=True
)
```

### Integration with Existing Projects

The package is designed to integrate seamlessly with existing data generation workflows:

```python
# Integration with tabularpriors
from tabpfn_v1_prior import TabPFNPriorDataLoader

# Use directly as a PyTorch DataLoader
dataloader = TabPFNPriorDataLoader(
    prior_type='mlp',
    num_steps=50,
    batch_size=16,
    num_datapoints_max=512,
    num_features=20,
    max_num_classes=10,
    device=torch.device('cuda'),
    flexible=True
)

for batch in dataloader:
    # Process batch
    pass
```

## API Reference

### `build_tabpfn_prior()`

Main factory function for creating TabPFN priors.

**Parameters:**
- `prior_type` (str): Type of prior ('mlp', 'gp', 'gp_mix', 'prior_bag')
- `num_steps` (int): Number of batches per epoch (default: 100)
- `batch_size` (int): Number of functions per batch (default: 8)
- `num_datapoints_max` (int): Maximum sequence length (default: 1024)
- `num_features` (int): Number of input features (default: 10)
- `max_num_classes` (int): Maximum number of classes (default: 10)
- `device` (str): Device to use ('cpu' or 'cuda', default: 'cpu')
- `prior_config` (dict): Custom hyperparameter configuration (optional)
- `flexible` (bool): Enable flexible categorical encoding (default: True)
- `differentiable` (bool): Enable differentiable hyperparameters (default: False)
- `return_categorical_mask` (bool): Return categorical feature mask when flexible=True (default: False)

**Returns:**
- `TabPFNPriorDataLoader`: Configured prior data loader

### `TabPFNPriorDataLoader`

PyTorch DataLoader for TabPFN prior generation.

**Batch Format:**
Each batch contains:
- `x`: Input features `[batch_size, seq_len, num_features]`
- `y`: Labels `[batch_size, seq_len]`
- `target_y`: Target labels (same as y) `[batch_size, seq_len]`
- `single_eval_pos`: Evaluation position (int)
- `categorical_mask`: Boolean tensor `[num_features]` (only when `return_categorical_mask=True`)

## File Structure

```
tabpfn-v1-prior/
│
├── tabpfn_prior/
│   ├── __init__.py
│   ├── tabpfn_prior.py
│   ├── test_standalone.py
│   └── priors/
│       ├── __init__.py
│       ├── differentiable_prior.py
│       ├── fast_gp.py
│       ├── flexible_caterogical.py
│       ├── mlp.py
│       ├── prior.py
│       ├── prior_bag.py
│       └── utils.py
│
├── README.md
└── pyproject.toml
```

## Examples

### Basic Data Generation

```python
import torch
from tabpfn_v1_prior import build_tabpfn_prior

# Generate classification data
prior = build_tabpfn_prior(
    prior_type='mlp',
    num_steps=10,
    batch_size=4,
    num_datapoints_max=100,
    num_features=5,
    max_num_classes=3,
    device='cpu'
)

for i, batch in enumerate(prior):
    print(f"Batch {i+1}:")
    print(f"  Features: {batch['x'].shape}")
    print(f"  Labels: {batch['y'].shape}")
    print(f"  Eval position: {batch['single_eval_pos']}")
    
    if i >= 2:  # Only show first 3 batches
        break
```

### Research Configuration

```python
# Advanced research setup with differentiable hyperparameters
research_prior = build_tabpfn_prior(
    prior_type='prior_bag',
    num_steps=50,
    batch_size=16,
    num_datapoints_max=512,
    num_features=20,
    max_num_classes=10,
    device='cuda',
    flexible=True,
    differentiable=True,
    prior_config={
        'verbose': True,
        'emsize': 512,  # Larger embedding for differentiable hyperparameters
    }
)
```

## License

This package contains code extracted and adapted from the original TabPFN project. Please refer to the original TabPFN license for usage terms.

## Citation

If you use this package in your research, please cite the original TabPFN paper:

```bibtex
@article{hollmann2022tabpfn,
  title={TabPFN: A Transformer that Solves Small Tabular Classification Problems in a Second},
  author={Hollmann, Noah and M{\"u}ller, Samuel and Eggensperger, Katharina and Hutter, Frank},
  journal={arXiv preprint arXiv:2207.01848},
  year={2022}
}
```

## Contributing

This is a standalone extraction of TabPFN's prior generation capabilities. For improvements to the core algorithms, please contribute to the original TabPFN project.

For issues specific to this standalone package, please ensure they are related to the packaging and integration aspects rather than the core prior algorithms.