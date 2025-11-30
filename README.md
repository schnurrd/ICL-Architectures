# ICL-Architectures
Unified framework for comparing model architectures in in-context learning across tabular and time-series domains. Includes modular pretraining pipelines, shared priors, and evaluations of Transformers, linear attention, xLSTM, Mamba, and SSM variants.


## Installation

```bash
conda create -n icl_arch_pfn python=3.10
conda activate icl_arch_pfn
pip install -r requirements.txt
```

### Install PFNs repo dependencies
```bash
pip install -e ./PFNs
```

### Install TabPFN-v1-prior repo dependencies
```bash
pip install -e prior-repos/tabpfn-v1-prior
```

### Install tabularpriors repo dependencies
```bash
pip install -e prior-repos/tabularpriors
```

Tested for Nvidia RTX 5070 with Cuda 12.8.

## Run pretraining

To run pre-training with the TabPFNv1 prior use:

```bash
python PFNs/pfns/run_training_cli.py PFNs/tabpfn_prior_config.py --device cuda:0 --checkpoint-save-load-prefix PFNs/models_diff/test.pt
```

## Credits
This repo builds on:
- [PFNs](https://github.com/automl/PFNs) (Apache 2.0) for the core training pipeline and priors.
- [TabPFN-v1-prior](https://github.com/automl/tabpfn-v1-prior) (Apache 2.0) for the tabpfn v1 prior implementation.
- [tabularpriors](https://github.com/automl/tabularpriors) (Apache 2.0) for additional tabular priors (TabICL, TICL)


# PFNs documentation

## CLI interface

The training CLI allows you to train PFNs models using configuration from Python files. This provides a flexible and programmable way to configure training parameters, allowing for dynamic configuration generation, conditional logic, and easy reuse of configuration components. Configuration files define a `config` variable containing the training configuration.

### Usage
```bash
python PFNs/pfns/run_training_cli.py PFNs/tabpfn_prior_config.py \
    --device cuda:0 \
    --compile \
    --checkpoint-save-load-prefix PFNs/models_diff/test.pt \
    --checkpoint-save-load-suffix no_seed \
    --tensorboard-path PFNs/tensorboards \
    --config-index 0
```

## Command Line Arguments

- `config_file` (required): Path to the Python configuration file that defines a `config` variable
- `--device`: Device to use for training (e.g., 'cuda:0', 'cpu'). If not specified, will auto-detect.
- `--compile`: Use torch.compile for the model (requires PyTorch 2.0+)
- `--checkpoint-save-load-prefix`: Path to save/load checkpoint and for tensorboard.
- `--checkpoint-save-load-suffix`: Suffix to add to the checkpoint save/load path. this can e.g. be the seed.
- `--tensorboard-path`: Path to save tensorboard. If not provided, will use the checkpoint save/load prefix or the path in the config file.
- `--config-index`: Index of the config to use. This is used to select a config from the config file.
