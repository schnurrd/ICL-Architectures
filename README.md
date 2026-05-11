# ICL-Architectures

Unified framework for comparing sequence-model architectures for in-context learning on tabular classification tasks. Includes modular pretraining pipelines, shared priors, and evaluations of Transformer, Linear Attention, DeltaNet, Gated DeltaNet, Kimi Delta Attention, and Mamba2 backbones.

## Table of Contents

- [Installation](#installation)
  - [Pulling latest changes](#pulling-latest-changes)
- [Repository User Guide](#repository-user-guide)
  - [CLI training interface](#cli-training-interface)
    - [Usage](#usage)
    - [Command Line Arguments](#command-line-arguments)
  - [CLI evaluation interface](#cli-evaluation-interface)
    - [Usage](#usage-1)
    - [Command Line Arguments](#command-line-arguments-1)
  - [Configuration and logging](#configuration-and-logging)
    - [wandb support](#wandb-support)
    - [Configuration Files](#configuration-files)
  - [Sequence-length experiments](#sequence-length-experiments)
  - [Minimal sequence-length degradation reproduction](#minimal-sequence-length-degradation-reproduction)
- [Repository (PFNs) explanation](#repository-pfns-explanation)
  - [Steps of execution in the pre-training pipeline](#steps-of-execution-in-the-pre-training-pipeline)
  - [Main Config components](#main-config-components)
  - [Model Overview](#model-overview)
    - [Encoding](#encoding)
      - [Features per group parameter](#features-per-group-parameter)
      - [Encoding overview](#encoding-overview)
      - [Feature Positional Embedding](#feature-positional-embedding)
    - [Backbones](#backbones)
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

Install the required packages and editable installs for PFNs and the TabPFN-v1 prior:

```bash
conda create -n icl_arch python=3.11
conda activate icl_arch

pip install -r requirements/requirements.txt \
    -e ./PFNs \
    -e ./prior-repos/tabpfn-v1-prior
```

Tested for Nvidia RTX 5070 with CUDA 12.8. For old GPUs with compute capability < 7.0 you might need to install `requirements/requirements_old_gpus.txt` instead (e.g. Tesla P100, Titan Xp, Titan X). Additionally, `torch.compile` will not work.

On the clusters with CUDA 11.8, the following versions work:

```bash
conda create -n icl_arch python=3.11
conda activate icl_arch

conda install -y -c nvidia/label/cuda-11.8.0 cuda-toolkit

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

python -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.7.1

python -m pip install --no-cache-dir --no-build-isolation causal-conv1d mamba-ssm

pip install -r requirements/requirements_obsession.txt \
    -e ./PFNs \
    -e ./prior-repos/tabpfn-v1-prior
```

### Pulling latest changes

To pull the latest changes including submodules, run:

```bash
git pull
git submodule update --init --recursive
```

Additionally to clean the submodules or main repository run: `git submodule foreach git clean -fdx` and `git clean -fdx` for main repository. To see the status of the submodules run: `git submodule status`.

# Repository User Guide

## CLI training interface

The training CLI allows you to train PFNs models using configuration from Python files. This provides a flexible and programmable way to configure training parameters, allowing for dynamic configuration generation, conditional logic, and easy reuse of configuration components. Configuration files define either a `config` variable or a `get_config(...)` function returning the training configuration.

### Usage

Transformer example:

```bash
python PFNs/pfns/run_training_cli.py PFNs/configs/transformer/transformer_config.py \
    --device cuda:0 \
    --compile \
    --checkpoint-save-load-prefix PFNs/models_diff \
    --checkpoint-save-load-suffix no_seed \
    --wandb \
    --config-index 0
```

FLA example:

```bash
python PFNs/pfns/run_training_cli.py PFNs/configs/fla/fla_config.py \
    --device cuda:0 \
    --checkpoint-save-load-prefix PFNs/models_diff \
    --checkpoint-save-load-suffix gla_seed0 \
    --config-arg model_type=gla \
    --config-arg sequence_mode=Comb_ST \
    --config-arg training_setup=high \
    --config-index 0
```

### Command Line Arguments

- `config_file` (required): Path to the Python configuration file that defines either a `config` variable or a `get_config(...)` function
- `--device`: Device to use for training (e.g., 'cuda:0', 'cpu'). If not specified, will auto-detect.
- `--compile`: Use torch.compile for the model (requires PyTorch 2.0+)
- `--checkpoint-save-load-prefix`: Path to save/load checkpoint (and default wandb dir).
- `--checkpoint-save-load-suffix`: Suffix to add to the checkpoint save/load path. this can e.g. be the seed.
- `--wandb` / `--no-wandb`: Enable/disable wandb logging (wandb settings come from the config file).
- `--continue-from-wandb`: Continue training from a wandb run path (`entity/project/run_id` or `project/runs/run_id`), downloading the checkpoint if needed.
- `--config-index`: Index of the config to use. This is used to select a config from the config file.
- `--config-arg`: Extra `get_config` keyword argument as `KEY=VALUE`; repeat for multiple overrides.
- `--overwrite`: Start fresh even if a checkpoint/config exists at the target path (do not load, overwrite on save).
- `--train-mixed-precision` / `--no-train-mixed-precision`: Override mixed precision on/off after loading the config/checkpoint.
- `--train-mixed-precision-dtype`: Override mixed precision dtype after loading the config/checkpoint (e.g. `auto`, `fp16`, `bf16`, `fp32`).

### CLI evaluation interface

The evaluation CLI allows you to evaluate trained PFNs/TabPFN-style models on OpenML benchmarks. It reports tabular classification metrics and can load either a local checkpoint directory or a wandb run.

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

- `--model_path`: Path to the trained model checkpoint directory. Provide this or `--wandb_run_id`.
- `--checkpoint_name`: Name of the checkpoint file within the model path (default: 'checkpoint.pt')
- `--wandb_run_id`: wandb run path or ID to download/evaluate instead of a local checkpoint path.
- `--device`: Device to use for evaluation (e.g., 'cuda:0', 'cpu'). Default: auto-detect
- `--benchmark`: Benchmark suite to evaluate on. Choices: 'opencc' (OpenML-CC18), 'openml_large_dataset', 'tabarena_full', 'tabarena_medium'. Default: 'opencc'
- `--max_samples`: Maximum number of samples per dataset (default: 1000)
- `--max_features`: Maximum number of features per dataset (default: 20)
- `--max_classes`: Maximum number of classes per dataset (default: 10)
- `--n_splits`: Number of cross-validation splits (default: 5)
- `--output`: Path to save results as CSV file. If not provided, results are only printed to console
- `--n_jobs`: Number of CPU cores passed to evaluation helpers where applicable. Default: 4
- `--batch_size_inference`: Batch size for TabPFN inference (default: 32). Lower values reduce GPU memory usage without affecting accuracy - useful for memory-constrained environments
- `--n_ensemble_configurations`: Number of ensemble configurations for TabPFN (default: 32)
- `--preprocess_transforms`: Preprocessing transforms to ensemble over (default: `none power robust`)
- `--sample_order_permutation`: Permute training sample order for each ensemble configuration
- `--fla_cache_chunk_size`: Chunk size for cache-backed inference when using an FLA backbone

## Configuration and logging

### wandb support

wandb is configured via `MainConfig.wandb`. The CLI can toggle logging via `--wandb` / `--no-wandb` or continue from an existing run via `--continue-from-wandb`.
For restricted environments, set `mode="offline"` in the config file (or `WANDB_MODE=offline`) and sync later with `wandb sync`.

### Configuration Files

The Python configuration file must define either a `config` variable or a
`get_config(config_index: int = 0)` function, which when called returns a
`MainConfig` object. An example configuration file can be found at
`PFNs/configs/example_config.py`; the main TabPFN-style transformer config is
`PFNs/configs/transformer/transformer_config.py`.

The main FLA config is `PFNs/configs/fla/fla_config.py`. It supports
`model_type` values such as `kda`, `gla`, `mamba2`, `deltanet`,
`gated_deltanet`, and `linear_attn`, plus sequence-mode,
bidirectional, state-passing, cache, categorical-feature, and mimetic-init
options via `--config-arg`.

#### Curriculum Learning Parameters

Curriculum learning is configured via `get_config(...)` arguments in configs such as
`PFNs/configs/transformer/transformer_config.py` and `PFNs/configs/fla/fla_config.py`.
From the CLI, pass these through repeatable `--config-arg KEY=VALUE` arguments.

##### Sequence-length stages

- `max_seq_len` (default: `1000`): Upper bound for sampled sequence length.
- `seq_len_stages` (default: `None`): Optional staged sequence-length settings by epoch.
  Entries are applied in order. After the last stage, sampling falls back to a fixed sequence length of `max_seq_len`.
  Every stage max sequence length must be `<= max_seq_len`.
  Supported formats:
  - `(end_epoch, stage_max_seq_len)`
  - `(end_epoch, stage_max_seq_len, eval_pos_split_pct_min, eval_pos_split_pct_max)`
  - `(end_epoch, stage_min_seq_len, stage_max_seq_len, seq_len_distribution)`
  - `(end_epoch, stage_min_seq_len, stage_max_seq_len, seq_len_distribution, eval_pos_split_pct_min, eval_pos_split_pct_max)`
    Where `seq_len_distribution` is one of:
  - `fixed`: use `stage_max_seq_len`.
  - `uniform`: sample integer sequence length uniformly in `[min_seq_len, max_seq_len]`.
  - `log_uniform`: sample sequence length log-uniformly in `[min_seq_len, max_seq_len]`.
  - `shifted_lognormal`: sample sequence length from a shifted log-normal distribution calibrated for long-sequence staged training.

Examples:

- `--config-arg max_seq_len=16000 --config-arg seq_len_stages='[(5, 2048), (20, 8192), (60, 16000)]'`
- `--config-arg max_seq_len=12000 --config-arg seq_len_stages='[(10, 4000, 80, 80), (30, 12000, 30, 90)]'`
  - First stage: fixed eval split at 80%.
  - Second stage: eval split sampled from 30%-90%.
- `--config-arg max_seq_len=64000 --config-arg seq_len_stages='[(150, 1000, 5000, "uniform"), (200, 5000, 64000, "log_uniform")]'`
  - First stage: sample seq len uniformly from 1k to 5k.
  - Second stage: sample seq len log-uniformly from 5k to 64k.

##### Eval-position split (global)

- `eval_pos_split_pct` (default: `None`): Global eval split in percent for `single_eval_pos`.
  - Scalar: fixed split, e.g. `80` means always 80% of sequence length.
  - Pair: range, e.g. `(30, 90)` means sampled uniformly between 30% and 90%.
- Stage-level split values in `seq_len_stages` override the global `eval_pos_split_pct` for those epochs.
- Split percentages must be between 0 and 100, with min `<=` max.

##### Dynamic batch-size by sequence length

- `batch_size_stages` (default: `None`): Optional sequence-length thresholds for batch size.
  - Format: `[(seq_len_threshold, batch_size), ...]` with increasing thresholds.
  - The first threshold `>= sampled_seq_len` is used.
  - If `sampled_seq_len` exceeds the largest threshold, the last configured batch size is used.
  - Example: `[(4096, 16), (16000, 8), (64000, 4)]`.
- `dynamic_batch_size_compensate_grad_accumulation` (default: `False`):
  - If enabled, each micro-batch contributes optimizer-step progress proportional to
    `dynamic_batch_size / base_batch_size`.
  - This keeps effective batch size approximately stable when dynamic batch sizing is active.

##### Related sampler controls

These are part of `BatchShapeSamplerConfig` and influence the same sampling process:

- `min_single_eval_pos`: Minimum position where evaluation targets start (`single_eval_pos` lower bound).
- `fixed_num_test_instances`: If set, enforces a fixed number of test items and derives final `seq_len` from `single_eval_pos + fixed_num_test_instances`.
- `min_num_features`, `max_num_features`: Feature-count sampling range per batch.
- `seed` (default: `42`): Seed used with `(epoch, step)` for deterministic batch-shape sampling.

## Sequence-length experiments

### Sequence-length notebooks

The main sequence-length experiment notebooks are:

- [Sequence-length comparison and generalization](PFNs/notebooks/seq_len_comparison_and_generalization.ipynb)
- [Minimal linear-attention sequence-length generalization](PFNs/notebooks/minimal_linear_attention_seq_len_generalization.ipynb)
- [Sequence-length hidden-state debugging](PFNs/notebooks/seq_len_hidden_state_debug.ipynb)

The corresponding batch script for the larger sequence-length benchmark is
[run_synthetic_seq_len_experiments.py](PFNs/notebooks/run_synthetic_seq_len_experiments.py).

## Minimal sequence-length degradation reproduction

The minimal reproduction for the linear-attention sequence-length degradation
question is the notebook
[minimal_linear_attention_seq_len_generalization.ipynb](PFNs/notebooks/minimal_linear_attention_seq_len_generalization.ipynb).
It trains small causal and non-causal linear-attention PFN variants on short
contexts and evaluates them on longer context lengths.

For non-interactive runs, the notebook has a script companion:
[minimal_linear_attention_seq_len_generalization.py](PFNs/notebooks/minimal_linear_attention_seq_len_generalization.py).
It writes outputs under `minimal_linear_attention_seq_len_generalization_runs/`
unless `--output-root` is set, and can log to the
`minimal_linear_attention_seq_len_generalization` wandb project with `--wandb`.

# Repository (PFNs) explanation

## Steps of execution in the pre-training pipeline

1. The CLI script `run_training_cli.py` is executed with the path to a configuration file and the CLI parameters. This first parses the CLI arguments and then loads the configuration file as a Python module. It retrieves the `config` variable or calls the `get_config` function to obtain the `MainConfig` object.
2. The `MainConfig` object includes all the necessary objects for the training process, including the prior, model, batch shape sampler, optimizer and training loop configuration. If we have already started a training with the same name and have stored a checkpoint the config gets updated to load the checkpoint.
3. The training loop is started by calling the `pfns.train.train` function with the created `MainConfig` object.

## Main Config components

Dataclass (see `PFNs/pfns/train.py`) that includes all necessary components for training. Specifically includes:

- Prior and optimizer:
  - **Prior**: prior.PriorConfig objects defining the prior
  - **Optimizer**: OptimizerConfig object defining the optimizer
- Model:
  - **model**: ModelConfig object defining the model architecture
- Training:
  - **batch_shape_sampler**: BatchShapeSamplerConfig object which samples num_features, and single_eval_pos for each batch
  - **epochs**: Number of training epochs
  - **steps_per_epoch**: Number of steps per epoch (we don't really have a concept of epochs since data is infinite, so this defines how many steps we call an epoch)
  - **aggregate_k_gradients**: Number of batches to aggregate gradients over before performing an optimizer step, allowing a larger effective batch size than fits in GPU memory
  - **n_targets_per_input**: Used if a model is trained to predict multiple targets per input
  - **train_mixed_precision**, **train_mixed_precision_dtype**
  - **skip_grad_norm_spike_factor**
- LR Scheduler:
  - **scheduler**, **warmup_epochs**, **min_lr**
- Checkpointing:
  - **train_state_dict_save_path**, **train_state_dict_load_path**
- Validation:
  - **test_priors**: prior.PriorConfig objects to use for validation during training
  - **validation_period**: How often to run validation (in epochs)
- Logging:
  - **verbose**, **progress_bar**, **wandb**, **wandb_run_id**: Logging options
- Data loading
  - **dataloader_class**, **num_workers**

## Model Overview

### Encoding

Encoders are a sequence of (learned) transformations (encoding steps) that process the input data (x and y) before into an embedding that is fed into the main sequence model (e.g. Transformer). Different encoding steps can be stacked to form the final encoder. The different encoders currently implemented are in `PFNs/pfns/model/encoders.py` and implement the abstract base class `SeqEncStep`:

- **ConstantNormalizationInputEncoderStep**: Input normalization with a provided mean and std.
- **InputNormalizationEncoderStep**: Performs simple outlier soft clipping using logarithmic compression and input normalization to mean 0 and std 1.
- **VariableNumFeaturesEncoderStep**: Transforms input to a fixed number of features by appending zeros and performs normalization by number of features used to keep variance consistent.
- **NanHandlingEncoderStep**: Creates NaN masks for input and target and replaces NaNs with feature mean.
- **OrdinalEncoderStep**: Encodes categorical feature values as ordinal IDs using the training context.
- **MixedFeatureEncoderStep**: Applies categorical or continuous preprocessing per feature based on categorical feature metadata.
- **LinearInputEncoderStep**: Linear layer to map input features to model dimension (num_features -> embedding size). Normally a single layer but with optional 2-layer MLP with GELU activation.

These individual encoders can be combined using the `SequentialEncoder` class to form a complete encoding pipeline.

**Style Encoder**: Special encoder that encodes metadata (e.g. hyperparameters) that describes how the data was generated, allowing the model to condition on this information.

#### Features per group parameter

The default encoding of the transformer model creates one embedding per feature. TabPFN v1 used one embedding per row (`features_per_group = num_features`). The larger `features_per_group` is set, the fewer tokens the sequence model has to process, reducing memory and compute requirements. However, this also reduces the model capacity.

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

### Backbones

The model core is selected via `ModelConfig.backbone`, which uses the `Backbone` / `BackboneConfig` interfaces in `PFNs/pfns/model/backbones.py`. Current backbone implementations include:

- **TransformerBackbone**: PFN-style per-feature Transformer stack.
- **FLABackbone** and **BidirectionalFLABackbone**: Wrappers for Flash Linear Attention models such as GLA, DeltaNet, Gated DeltaNet, KDA, Mamba2, Linear Attention, and MesaNet.
- **LinearAttentionBackbone** and related experimental backbones for local linear-attention variants.

### Table Transformer Architecture

The Transformer backbone extends the standard Transformer architecture to operate on a per-feature basis, which allows for processing each feature separately. It only consists of encoder blocks (no decoder blocks). Specifically, the Transformer backbone is a stack of `PerFeatureLayer` layers.

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

`TabPFNClassifier` is the main prediction interface with a modular architecture supporting swappable model backbones. The training set is provided via `fit(X_train, y_train)`, and predictions are made on new data with `predict(X_test)` or `predict_proba(X_test)`.

### Architecture Components

The inference pipeline consists of three main components:

1. **EnsembleConfig**: Dataclass defining configuration for each ensemble member (class shift, feature shift, preprocessing transform type)

2. **InferenceEngine**: Handles preprocessing, ensemble prediction, and aggregation
   - Manages preprocessing transformations (PowerTransformer, QuantileTransformer, RobustScaler)
   - Applies class/feature shifts for ensemble diversity
   - Batches inference across ensemble members and aggregates results

3. **Backbone interface**: Swappable neural architecture interface implemented in `PFNs/pfns/model/backbones.py`
   - `TransformerBackbone`: Wrapper for the PFN-style Transformer stack
   - `FLABackbone`: Wrapper for Flash Linear Attention backbones used by GLA, DeltaNet, KDA, Mamba2, Linear Attention, MesaNet, and related variants

### Prediction Flow

During the `fit` call, the training data is preprocessed:

- Y values are encoded via sklearn's LabelEncoder (outputs 0,...,n_classes-1)
- X and y are stored for use during prediction (no actual training occurs here)

During `predict`/`predict_proba` calls:

1. **Generate ensemble configurations**: Create N configurations with random class/feature shifts and preprocessing transforms
2. **Preprocess data**: InferenceEngine applies transforms (with caching by transform type), removes constant features, handles outliers
3. **Apply shifts**: Circular shift classes/features according to each configuration
4. **Batch inference**: Forward passes through model backbone in batches
5. **Aggregate**: Reverse class shifts, average logits across ensemble, apply softmax

# Credits

This repo builds on:

- [PFNs](https://github.com/automl/PFNs) (Apache 2.0) for the core training pipeline and priors. Used as the starting repository.
- [TabPFN-v1-prior](https://github.com/automl/tabpfn-v1-prior) (Apache 2.0) for the tabpfn v1 prior implementation.

# Similar relevant repositories

- [TabPFN](https://github.com/PriorLabs/TabPFN) the TabPFN model and prior implementation.
- [TFM-Playground](https://github.com/automl/TFM-Playground) (Apache 2.0) similar to this repository however still in initial stages.
- [nanoTabPFN](https://github.com/automl/nanoTabPFN) small educational version of TabPFN.
- [TabICL](https://github.com/soda-inria/tabicl) For TabICL model and prior implementation from Inria.
