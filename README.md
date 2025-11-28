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
