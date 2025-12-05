# ICL-Architectures
Unified framework for comparing model architectures in in-context learning across tabular and time-series domains. Includes modular pretraining pipelines, shared priors, and evaluations of Transformers, linear attention, xLSTM, Mamba, and SSM variants.

## Table of Contents
- [Installation](#installation)
- [Repository User Guide](#repository-user-guide)
  - [CLI training interface](#cli-training-interface)
    - [Usage](#usage)
    - [Command Line Arguments](#command-line-arguments)
  - [CLI evaluation interface](#cli-evaluation-interface)
    - [Usage](#usage-1)
    - [Command Line Arguments](#command-line-arguments-1)
  - [Tensorboard support](#tensorboard-support)
  - [Configuration Files](#configuration-files)
- [Currently known issues / TODOs](#currently-known-issues--todos)
- [Repository (PFNs) explanation](#repository-pfns-explanation)
  - [Steps of execution in the pre-training pipeline](#steps-of-execution-in-the-pre-training-pipeline)
  - [Main Config components](#main-config-components)
  - [Model Overview](#model-overview)
    - [Encoding](#encoding)
      - [Features per group parameter](#features-per-group-parameter)
      - [Encoding overview](#encoding-overview)
      - [Feature Positional Embedding](#feature-positional-embedding)
    - [Table Transformer Architecture](#table-transformer-architecture)
      - [Per Feature Layer](#per-feature-layer)
    - [Decoder](#decoder)
  - [Inference Overview](#inference-overview)
- [Credits](#credits)
- [Similar relevant repositories](#similar-relevant-repositories)

## Installation

Clone the repository with submodules:

```bash
git clone --recurse-submodules git@github.com:schnurrd/ICL-Architectures.git
cd ICL-Architectures
```

Install the required packages and editable installs for the repositories PFNs, TabPFN-v1-prior, and tabularpriors:

```bash
conda create -n icl_arch_pfn python=3.10
conda activate icl_arch_pfn

pip install -r requirements.txt \
    -e ./PFNs \
    -e ./prior-repos/tabpfn-v1-prior \
    -e ./prior-repos/tabularpriors
```

Tested for Nvidia RTX 5070 with Cuda 12.8. For old GPUs with compute capability < 7.0 you might need to install requirements_old_gpu.txt instead (e.g. Tesla P100, Titan Xp, Titan X) (TODO currently this still does not work).

# Repository User Guide

## CLI training interface

The training CLI allows you to train PFNs models using configuration from Python files. This provides a flexible and programmable way to configure training parameters, allowing for dynamic configuration generation, conditional logic, and easy reuse of configuration components. Configuration files define a `config` variable containing the training configuration.

### Usage
```bash
python PFNs/pfns/run_training_cli.py PFNs/configs/tabpfn_prior_config.py \
    --device cuda:0 \
    --compile \
    --checkpoint-save-load-prefix PFNs/models_diff/test.pt \
    --checkpoint-save-load-suffix no_seed \
    --tensorboard-path PFNs/tensorboards \
    --config-index 0
```

or Multiple GPUs (e.g. 2 GPUs):

```bash
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 PFNs/pfns/run_training_cli.py \
    PFNs/configs/tabpfn_prior_config_very_large_2_gpu.py \
    --checkpoint-save-load-prefix PFNs/models_diff/large_config_2_gpu.pt \
    --checkpoint-save-load-suffix no_seed \
    --tensorboard-path PFNs/tensorboards \
    --config-index 0
```


### Command Line Arguments

- `config_file` (required): Path to the Python configuration file that defines a `config` variable
- `--device`: Device to use for training (e.g., 'cuda:0', 'cpu'). If not specified, will auto-detect.
- `--compile`: Use torch.compile for the model (requires PyTorch 2.0+)
- `--checkpoint-save-load-prefix`: Path to save/load checkpoint and for tensorboard.
- `--checkpoint-save-load-suffix`: Suffix to add to the checkpoint save/load path. this can e.g. be the seed.
- `--tensorboard-path`: Path to save tensorboard. If not provided, will use the checkpoint save/load prefix or the path in the config file.
- `--config-index`: Index of the config to use. This is used to select a config from the config file.

### CLI evaluation interface

The evaluation CLI allows you to evaluate trained PFNs models against baselines (RandomForest, XGBoost) on OpenML benchmarks. This provides a standardized way to assess model performance on tabular classification tasks.

#### Usage
```bash
python PFNs/pfns/run_evaluation_cli.py \
    --model_path PFNs/models_diff/large_config.pt/tabpfn_prior_config_large_0_no_seed \
    --benchmark opencc \
    --n_splits 5 \
    --output results.csv \
    --batch_size_inference 16
```

#### Command Line Arguments

- `--model_path` (required): Path to the trained model checkpoint directory
- `--checkpoint_name`: Name of the checkpoint file within the model path (default: 'checkpoint.pt')
- `--device`: Device to use for evaluation (e.g., 'cpu', 'cpu'). Default: auto-detect
- `--benchmark`: Benchmark suite to evaluate on. Choices: 'opencc' (OpenML-CC18), 'test' (smaller test set). Default: 'opencc'
- `--max_samples`: Maximum number of samples per dataset (default: 1024)
- `--max_features`: Maximum number of features per dataset (default: 25)
- `--max_classes`: Maximum number of classes per dataset (default: 10)
- `--n_splits`: Number of cross-validation splits (default: 5)
- `--output`: Path to save results as CSV file. If not provided, results are only printed to console
- `--only_tabpfn`: Flag to evaluate only TabPFN without baseline comparisons
- `--n_jobs`: Number of CPU cores for baseline models (RandomForest, XGBoost). Default: 4. Use this to limit CPU usage on shared machines
- `--batch_size_inference`: Batch size for TabPFN inference (default: 32). Lower values reduce GPU memory usage without affecting accuracy - useful for memory-constrained environments

## Tensorboard support

Tensorboard can be added via the `tensorboard_path` CLI parameter or by setting it in the `MainConfig`. The training logs can then be viewed by starting the tensorboard with: 

```bash
tensorboard --logdir TENSORBOARD_PATH
```

## Configuration Files  

The Python configuration file must define a `config`or a `get_config(config_index: int = 0)` function, which when called returns a `MainConfig` object. An example configuration file can be found at `PFNs/tabpfn_prior_config.py`.

# Currently known issues / TODOs

- Old GPUs with compute capability < 7.0 (e.g. Tesla P100, Titan Xp, Titan X) do not work properly due to Cuda version incompatibilities. Please use a newer GPU if possible
- Multi GPU training currently does not provide the expected predictive performance and is worse than single GPU training
- Samplers used are very basic and could be improved to better cover the data distribution
- Look into replacing the Inference wrapper with the prior labs tabpfn implementation

# Repository (PFNs) explanation

## Steps of execution in the pre-training pipeline
1. The CLI script `run_training_cli.py` is executed with the path to a configuration file and the CLI parameters. This first parses the CLI arguments and then loads the configuration file as a Python module. It retrieves the `config` variable or calls the `get_config` function to obtain the `MainConfig` object.
2. The `MainConfig` object includes all the necessary objects for the training process, including the prior, model, batch shape sampler, optimizer and training loop configuration. If we have already started a training with the same name and have stored a checkpoint the config gets updated to load the checkpoint.
3. The training loop is started by calling the `pfns.train.train` function with the created `MainConfig` object.

## Main Config components

Dataclass (see `PFNs/pfns/train.py`) that includes all necessary components for training. Specifically includes:
- Training:
    - **Prior**: prior.PriorConfig objects defining the prior
    - **Optimizer**: OptimizerConfig object defining the optimizer
- Model:
    - **model**: TransformerConfig object defining the model architecture
- Training:
    - **batch_shape_sampler**: BatchShapeSamplerConfig object which samples num_features, and single_eval_pos for each batch
    - **epochs**: Number of training epochs
    - **steps_per_epoch**: Number of steps per epoch (we don't really have a concept of epochs since data is infinite, so this defines how many steps we call an epoch)
    - **aggregate_k_gradients**: Number of batches to aggregate gradients over before performing an optimizer step, allows an larger effective batch size than fits in GPU memory
    - **n_targets_per_input**: Used if a model is trained to predict multiple targets per input
    - **train_mixed_precision**
- LR Scheduler:
    - **scheduler**, **warmup_epochs**
- Checkpointing:
    - **train_state_dict_save_path**, **train_state_dict_load_path**
- Validation: 
    - **test_priors**: prior.PriorConfig objects to use for validation during training
    - **validation_period**: How often to run validation (in epochs)
- Logging:
    - **verbose**, **progress_bar**, **tensorboard_path**: Logging options
- Data loading
    - **dataloader_class**, **num_workers**

## Model Overview

### Encoding

Encoders are a sequence of (learned) transformations (encoding steps) that process the input data (x and y) before into an embedding that is fed into the main sequence model (e.g. Transformer). Different encoding steps can be stacked to form the final encoder. The different encoders currently implemented are in `PFNs/pfns/model/encoders.py` and implement the abstract base class `SeqEncStep`:

- **Constant Normalization Input Encoder**: Input normalization with a provided mean and std.
- **InputNormalizationEncoder**: Performs simple outlier soft clipping using logarithmic compression and input normalization to mean 0 and std 1.
- **VariableNumFeaturesEncoder**: Transforms input to a fixed number of features by appending zeros and performs normalization buy number of features used to keep variance consistent.
- **NanHandlingEncoder**: Creates NaN masks for input and target and replaces NaNs with feature mean.
- **LinearInputEncoder**: Linear layer to map input features to model dimension (num_features -> embedding size). Normally single layer but with optinal 2-layer MLP with GELU activation.

These individual encoders can be combined using the `SequentialEncoder` class to form a complete encoding pipeline.

**Style Encoder**: Special encoder that encodes metadata (e.g. hyperparameters) that describes how the data was generated, allowing the model to condition on this information.

#### Features per group parameter

The default encoding of the transformer model creates one embedding per features. TabPFN v1 used one embedding per row (features_per_group = num_features). The larger the feature_per_group is set the less tokens the sequence model has to process, reducing memory and compute requirements. However, this also reduces the model capacity.

#### Encoding overview

The model both encodes the input features (X) and the target (y) separately. Each of which goes through its own pipeline and has its own learned parameters.

#### Feature Positional Embedding

Without positional embeddings, the model cannot distinguish between different feature groups as attention is permutation-invariant. The `feature_positional_embedding` adds a unique identifier to each feature group's embedding. Options are: 
- `None`: No embedding (features indistinguishable)
- `normal_rand_vec`: Random Gaussian vectors (fixed per seed)
- `uni_rand_vec`: Random uniform [-1, 1] vectors
- `learned`: Lookup table of 1000 learnable embeddings
- `subspace` (recommended): Random vectors projected through a learned linear layer — combines random uniqueness with learnable representations

The random embeddings use a fixed seed, ensuring consistent feature IDs across forward passes while different models get different IDs. 

### Table Transformer Architecture

Extends the standard Transformer architecture to operate on a per-feature basis, which allows for processing each feature separately. The transformer here only consists of encoder blocks (no decoder blocks). Specifically the transformer is a stack of `PerFeatureLayer` layers.

#### Per Feature Layer

Transformer encoder layer that processes each feature block separately. Does Multi-head attention between features, multi-head attention between items, and feedforward neural networks (MLPs).

**Architecture flow:**
1. **Attention between features** (optional): Each feature group attends to other feature groups. Operates independently for each item in the sequence.
2. **Second MLP** (optional): Extra feedforward between attention layers  
3. **Attention between items**: Items/samples attend to other items in the sequence. Each feature group attends to itself across items.
4. **MLP**: Standard feedforward network

Each sublayer is followed by layer normalization (post-norm).

The main features here are:
- **Two attention axes**: Attends across both features AND items (unlike standard transformers)
- **Train/test split**: `single_eval_pos` separates training context from test items, enabling causal masking for in-context learning

### Decoder

The decoder is a simple MLP output head, that maps the y-token embeddings from test items to prediction logits:

1. After the transformer layers, extract the **y-token embedding** (last token in feature dimension) for each **test item**
2. Pass through MLP: `Linear(d_model → nhid) → GELU → Linear(nhid → n_outputs)`

The y-token for test items initially encodes "unknown label" (via NaN handling), and through attention with training context, accumulates information needed for prediction.

## Inference Overview

`TabPFNClassifier` is the main prediction interface. The training set is provided via `fit(X_train, y_train)`, and predictions are made on new data with `predict(X_test)` or `predict_proba(X_test)`. 

During the `fit` call, the training data is preprocessed:
- Y values are encoded via sklearn's LabelEncoder (outputs 0,...,n_classes-1)
- X and y are stored for use during prediction (no actual training occurs here)

During `predict`/`predict_proba` calls the following then happens:
- y_test labels are initiated to zeros
- Call into `transformer_predict` function 

`transformer_predict` is currently the main prediction method that handles both preprocessing and model inference. It performs the following steps:
1. Build iterator of possible preprocessings:
    - Shift classes and features according to a random permuation
    - Apply preprocessing transforms (e.g. power transform or none)
2. Iterate over preprocessing configurations:
    - Apply preprocessing to X_train and X_test
    - Store preprocessed versions along
3. Iterate over preprocessed datasets in batches:
    - Predict the targets for each batch i.e. one forward pass through the model
    - Store the predicted logits
4. Iterate over the preprocessing configurations
    - Reverse the class shifting
    - Average the logits over all preprocessing configurations


# Credits
This repo builds on:
- [PFNs](https://github.com/automl/PFNs) (Apache 2.0) for the core training pipeline and priors. Used as the starting repository.
- [TabPFN-v1-prior](https://github.com/automl/tabpfn-v1-prior) (Apache 2.0) for the tabpfn v1 prior implementation.
- [tabularpriors](https://github.com/automl/tabularpriors) (Apache 2.0) for additional tabular priors (TabICL, TICL)

# Similar relevant repositories
- [TabPFN](https://github.com/PriorLabs/TabPFN) the TabPFN model and prior implementation.
- [TFM-Playground](https://github.com/automl/TFM-Playground) (Apache 2.0) similar to this repository however still in initial stages.
- [nanoTabPFN](https://github.com/automl/nanoTabPFN) small educational version of TabPFN.
- [TabICL](https://github.com/soda-inria/tabicl) For TabICL model and prior implementation from Inria.
- [TICL](https://github.com/microsoft/ticl) For TICL model and prior implementation from Microsoft.