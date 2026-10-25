# Adapting Linear-Time Architectures for Tabular In-Context Learning

David Schnurr, Felix Sarnthein, Thomas Hofmann, Imanol Schlag

[Paper (arXiv)](https://arxiv.org/abs/2609.36337)

Unified framework for comparing sequence-model architectures for in-context learning on tabular classification tasks. Includes modular pretraining pipelines, shared priors, and evaluations of Transformer, (Gated) Linear Attention, (Gated) DeltaNet, and Mamba-2 backbones.

## Installation

Clone the repository with submodules. The TabPFN-v1 prior is included as a submodule in `prior-repos/tabpfn-v1-prior`.

```bash
git clone --recurse-submodules https://github.com/schnurrd/ICL-Architectures.git
cd ICL-Architectures
```

If the repository was cloned without `--recurse-submodules`, or after pulling changes, run `git submodule update --init --recursive`.

Dependencies are managed with [uv](https://docs.astral.sh/uv/getting-started/installation/). Create the environment and activate it:

```bash
uv sync
source .venv/bin/activate
```

Tested on Linux with NVIDIA RTX 5070 and RTX 2080 Ti GPUs and CUDA 12.8 and 12.9. Run the tests with `pytest`.

## Training

Models are trained with `run_training_cli.py` from a Python config file that defines a `config` variable or a `get_config(...)` function.

Transformer example:

```bash
python PFNs/pfns/run_training_cli.py PFNs/configs/transformer/transformer_config.py \
    --checkpoint-save-load-prefix PFNs/models_diff \
    --checkpoint-save-load-suffix no_seed \
    --device cuda:0 \
    --no-wandb \
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
    --no-wandb \
    --config-index 0
```

The first training steps of an FLA model can take a few minutes while Triton compiles and autotunes its kernels; later runs reuse the cache.

All models from the paper are trained with the config files in `PFNs/configs/`. Variants such as final-state readout, the write-rate decay, the training setup and the mitigation strategies are selected with repeatable `--config-arg KEY=VALUE` arguments, using the arguments of each config's `get_config`. The configs and their curriculum-learning options are described in [PFNs/configs/README.md](PFNs/configs/README.md). Run the script with `--help` for all options, such as resuming from a W&B run with `--continue-from-wandb`.

W&B logging is on by default and goes to the default entity of the logged-in wandb user; set `WANDB_ENTITY` to log to a team instead, or pass `--no-wandb` to train without a wandb account. For restricted environments, set `mode="offline"` in the config file (or `WANDB_MODE=offline`) and sync later with `wandb sync`.

## Evaluation

`run_evaluation_cli.py` evaluates a trained model on OpenML benchmarks, from a local checkpoint directory or a wandb run (`--wandb_run_id`):

```bash
python PFNs/pfns/run_evaluation_cli.py \
    --model_path PFNs/models_diff/transformer_config_0_no_seed_default \
    --benchmark opencc \
    --n_splits 5 \
    --output results.csv \
    --batch_size_inference 16
```

This `--model_path` matches the transformer training example above when run with `--no-wandb`. If wandb is enabled, replace `default` with the wandb run ID printed by the training command.

Available benchmarks are `opencc` (OpenML-CC18, default), `openml_large_dataset`, `tabarena_full`, `tabarena_medium` and `tabarena_largest_5`. A lower `--batch_size_inference` reduces GPU memory use without changing the results. Run the script with `--help` for all options.

## Reproducing the paper

- [Sequence-length comparison and generalization](PFNs/notebooks/seq_len_comparison_and_generalization.ipynb)
- [Real-world experiments](PFNs/notebooks/real_world_experiments.ipynb)
- [Minimal linear-attention sequence-length generalization](PFNs/notebooks/minimal_linear_attention_seq_len_generalization.ipynb)
- [Sequence-length hidden-state debugging](PFNs/notebooks/seq_len_hidden_state_debug.ipynb)
- [DeltaNet effective write rates](PFNs/notebooks/deltanet_effective_beta_plots.ipynb)

The benchmarks themselves are run with [run_synthetic_seq_len_experiments.py](PFNs/notebooks/run_synthetic_seq_len_experiments.py) and [run_real_world_experiments.py](PFNs/notebooks/run_real_world_experiments.py); [report_oracle_real_data.py](PFNs/notebooks/report_oracle_real_data.py) summarises the hidden-state oracle on the largest TabArena datasets.

All notebooks except the minimal reproduction require trained model checkpoints. These checkpoints are not included in this release; train the models with the [training CLI](#training). To use a local checkpoint, replace the `wandb_run_id` of its entry in [model_registry.py](PFNs/pfns/experiments/model_benchmarks/model_registry.py) with `"base_path": "<directory containing checkpoint.pt>"`. Without a W&B login the notebooks skip the W&B results cache and compute all results locally.

### Minimal sequence-length degradation reproduction

The [minimal reproduction notebook](PFNs/notebooks/minimal_linear_attention_seq_len_generalization.ipynb) is standalone: it trains small causal and non-causal linear-attention PFN variants on short contexts and evaluates them on longer context lengths. At the default settings (two models, 100k steps each), a full run takes about 2 hours on an RTX 5070.

For non-interactive runs, the notebook has a script companion:
[minimal_linear_attention_seq_len_generalization.py](PFNs/notebooks/minimal_linear_attention_seq_len_generalization.py).
It writes outputs under `minimal_linear_attention_seq_len_generalization_runs/`
unless `--output-root` is set, and can log to the
`minimal_linear_attention_seq_len_generalization` wandb project with `--wandb`.

## Repository structure

| Path | Contents |
| --- | --- |
| `PFNs/pfns/` | The `pfns` package: training loop, training and evaluation CLIs, models, priors and experiment code |
| `PFNs/configs/` | Training configs for all models in the paper |
| `PFNs/notebooks/` | Analysis notebooks and benchmark scripts |
| `PFNs/tests/` | Test suite |
| `prior-repos/tabpfn-v1-prior/` | TabPFN-v1 prior (submodule) |

The model architecture is described in the [paper](https://arxiv.org/abs/2609.36337); the backbones are implemented in `PFNs/pfns/model/backbones.py`.

## Citation

```bibtex
@article{schnurr2026adapting,
  title   = {Adapting Linear-Time Architectures for Tabular In-Context Learning},
  author  = {Schnurr, David and Sarnthein, Felix and Hofmann, Thomas and Schlag, Imanol},
  journal = {arXiv preprint arXiv:2609.36337},
  year    = {2026}
}
```

## Credits

This repo builds on:

- [PFNs](https://github.com/SamuelGabriel/PFNs) (Apache 2.0) for the core training pipeline and priors. Used as the starting repository.
- [TabPFN-v1-prior](https://github.com/automl/tabpfn-v1-prior) (Apache 2.0) for the TabPFN v1 prior implementation.

## Similar relevant repositories

- [TabPFN](https://github.com/PriorLabs/TabPFN) the TabPFN model and prior implementation.
- [TFM-Playground](https://github.com/automl/TFM-Playground) (Apache 2.0) open source playground containing nanoTabPFN with more diverse prior support.
- [nanoTabPFN](https://github.com/automl/nanoTabPFN) small educational version of TabPFN.
- [TabICL](https://github.com/soda-inria/tabicl) For the TabICL model and prior implementation from Inria.
