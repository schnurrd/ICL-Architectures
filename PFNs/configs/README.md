# Training configurations

Each configuration file defines either a `config` variable or a
`get_config(config_index: int = 0, ...)` function that returns a `MainConfig` object.
Keyword arguments of `get_config` are set from the training CLI with repeatable
`--config-arg KEY=VALUE` arguments.

- `transformer/transformer_config.py`: the main TabPFN-style transformer config.
- `fla/fla_config.py`: the main FLA config. It supports `model_type` values `gla`, `mamba2`,
  `deltanet`, `gated_deltanet`, and `linear_attn`, plus options for the sequence mode,
  bidirectionality, final-state readout, the DeltaNet write-rate decay, state weaving, state
  passing, cache chunking, categorical features, and mimetic initialisation.
- `linear_attention/linear_attention_config.py`: the local linear-attention implementation,
  including the layer-local ridge state.
- `two_axis/two_axis_config.py`: TabPFN-v2-style two-axis processing.
- `transformer_experiments/tabpfn_transformer_masking.py`: transformer attention-masking variants
  (see [masking_doc.md](transformer_experiments/masking_doc.md)).

## Curriculum learning parameters

Curriculum learning is configured via `get_config(...)` arguments in configs such as
`transformer/transformer_config.py` and `fla/fla_config.py`.

### Sequence-length stages

- `max_seq_len` (default: `1000`): Upper bound for sampled sequence length.
- `seq_len_stages` (default: `None`): Optional staged sequence-length settings by epoch.
  Entries are applied in order. After the last stage, sampling falls back to a fixed sequence length of `max_seq_len`.
  Every stage max sequence length must be `<= max_seq_len`.
  Supported formats:
  - `(end_epoch, stage_max_seq_len)`
  - `(end_epoch, stage_max_seq_len, eval_pos_split_pct_min, eval_pos_split_pct_max)`
  - `(end_epoch, stage_min_seq_len, stage_max_seq_len, seq_len_distribution)`
  - `(end_epoch, stage_min_seq_len, stage_max_seq_len, seq_len_distribution, eval_pos_split_pct_min, eval_pos_split_pct_max)`

  `seq_len_distribution` is one of:
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

### Eval-position split (global)

- `eval_pos_split_pct` (default: `None`): Global eval split in percent for `single_eval_pos`.
  - Scalar: fixed split, e.g. `80` means always 80% of sequence length.
  - Pair: range, e.g. `(30, 90)` means sampled uniformly between 30% and 90%.
- Stage-level split values in `seq_len_stages` override the global `eval_pos_split_pct` for those epochs.
- Split percentages must be between 0 and 100, with min `<=` max.

### Dynamic batch-size by sequence length

- `batch_size_stages` (default: `None`): Optional sequence-length thresholds for batch size.
  - Format: `[(seq_len_threshold, batch_size), ...]` with increasing thresholds.
  - The first threshold `>= sampled_seq_len` is used.
  - If `sampled_seq_len` exceeds the largest threshold, the last configured batch size is used.
  - Example: `[(4096, 16), (16000, 8), (64000, 4)]`.
- `dynamic_batch_size_compensate_grad_accumulation` (default: `False`):
  - If enabled, each micro-batch contributes optimizer-step progress proportional to
    `dynamic_batch_size / base_batch_size`.
  - This keeps effective batch size approximately stable when dynamic batch sizing is active.

### Related sampler controls

These are part of `BatchShapeSamplerConfig` and influence the same sampling process:

- `min_single_eval_pos`: Minimum position where evaluation targets start (`single_eval_pos` lower bound).
- `fixed_num_test_instances`: If set, enforces a fixed number of test items and derives final `seq_len` from `single_eval_pos + fixed_num_test_instances`.
- `min_num_features`, `max_num_features`: Feature-count sampling range per batch.
- `seed` (default: `42`): Seed used with `(epoch, step)` for deterministic batch-shape sampling.
