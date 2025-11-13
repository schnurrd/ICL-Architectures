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