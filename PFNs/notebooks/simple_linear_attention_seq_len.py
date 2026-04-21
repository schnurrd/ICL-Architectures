from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from pfns.experiments.model_benchmarks.benchmark_batch_generators import _set_data_generation_seed as seed_everything
from pfns.experiments.model_benchmarks.plotting import build_model_style_map, plot_curves_from_df
from pfns.model.backbones import LinearAttentionBackboneConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_BF16 = DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

SAVE_DIR = Path('trained_simple_iid_linear_attention')
CHECKPOINT_DIR = SAVE_DIR / 'model_checkpoints'

SEED = 0

# Model and training hyperparameters
NUM_FEATURES = 4
MAX_TRAIN_CONTEXT_LEN = 256
TEST_LEN = 100
TRAIN_STEPS = 30_000
BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 12
NUM_HEADS = 4
LR = 3e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0
FORCE_RETRAIN = False
COMPILE_MODEL = False
LOG_EVERY = 0

# Prior distribution hyperparameters
PRIOR_MLP_HIDDEN_SIZE = 32
PRIOR_MLP_MAX_HIDDEN_LAYERS = 5
PRIOR_ACTIVATIONS = ('tanh', 'relu', 'gelu')

# Evaluation config
EVAL_CONTEXT_LENGTHS = (128, 256, 512, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000)
EVAL_BATCH_SIZE = 8
EVAL_BATCHES = 100

ACTIVATION_MAP = {
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'gelu': nn.GELU,
}

PRIOR_ACTIVATION_MODULES = tuple(ACTIVATION_MAP[name] for name in PRIOR_ACTIVATIONS)

class PriorMLP(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        num_hidden_layers: int,
        activation_module: type[nn.Module],
    ):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError('num_hidden_layers must be >= 1')

        layers: list[nn.Module] = []
        in_features = num_features
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(activation_module())
            in_features = hidden_size
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

def sample_latent_tasks(
    batch_size: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> list[PriorMLP]:
    depths = torch.randint(
        1,
        PRIOR_MLP_MAX_HIDDEN_LAYERS + 1,
        (batch_size,),
        device=device,
        generator=generator,
    )
    activation_ids = torch.randint(
        0,
        len(PRIOR_ACTIVATION_MODULES),
        (batch_size,),
        device=device,
        generator=generator,
    )

    init_seed = int(
        torch.randint(0, 2**31 - 1, (), device=device, generator=generator).item()
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        models = []
        for depth, activation_id in zip(depths.tolist(), activation_ids.tolist()):
            model = PriorMLP(
                NUM_FEATURES,
                PRIOR_MLP_HIDDEN_SIZE,
                num_hidden_layers=depth,
                activation_module=PRIOR_ACTIVATION_MODULES[activation_id],
            )
            models.append(model.to(device=device, dtype=DTYPE).requires_grad_(False))
    return models
def sample_examples_for_tasks(
    task_models: list[PriorMLP],
    seq_len: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not task_models:
        raise ValueError('task_models must not be empty')
    batch_size = len(task_models)
    x = torch.randn(
        batch_size,
        seq_len,
        NUM_FEATURES,
        device=DEVICE,
        dtype=DTYPE,
        generator=generator,
    )
    logits = torch.stack([model(x_i) for model, x_i in zip(task_models, x)])
    # ensure that we are not to much unbalanced by construction 
    balance_quantiles = torch.empty(
        batch_size,
        1,
        device=DEVICE,
        dtype=logits.dtype,
    ).uniform_(0.4, 0.6, generator=generator)
    sorted_logits = logits.float().sort(dim=1).values
    threshold_indices = (balance_quantiles * (seq_len - 1)).round().long()
    thresholds = sorted_logits.gather(1, threshold_indices).to(logits.dtype)
    y = (logits > thresholds).to(DTYPE)
    return x, y


class SimpleLinearAttentionPFN(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.x_encoder = nn.Linear(NUM_FEATURES, HIDDEN_SIZE)
        self.y_encoder = nn.Linear(1, HIDDEN_SIZE)
        self.backbone = backbone
        self.decoder = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE * 2),
            nn.GELU(),
            nn.Linear(HIDDEN_SIZE * 2, 2),
        )

    def forward(
        self,
        *,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        context_len = context_x.shape[1]
        x = torch.cat([context_x, query_x], dim=1)
        query_y = torch.zeros(
            query_x.shape[0],
            query_x.shape[1],
            device=query_x.device,
            dtype=query_x.dtype,
        )
        y = torch.cat([context_y, query_y], dim=1).unsqueeze(-1)

        embedded_x = self.x_encoder(x)
        embedded_y = self.y_encoder(y)
        embedded_input = (embedded_x + embedded_y).unsqueeze(2)

        encoded = self.backbone(
            embedded_input,
            single_eval_pos=context_len,
            rope_pairwise_positions=False,
        )
        return self.decoder(encoded[:, context_len:, -1])


def make_linear_attention_model(*, causal_train_only: bool, device: torch.device) -> SimpleLinearAttentionPFN:
    layer_kwargs: dict[str, Any] = {
        'causal_train_only': causal_train_only,
        'feature_map': 'elu',
        'norm_type': 'layernorm',
        'use_k_sum_normalization': False,
    }
    backbone = LinearAttentionBackboneConfig(
        nlayers=NUM_LAYERS,
        nhead=NUM_HEADS,
        mlp_hidden_dim=HIDDEN_SIZE * 2,
        use_final_norm=True,
        layer_kwargs=layer_kwargs,
    ).create_backbone(
        ninp=HIDDEN_SIZE,
        attention_between_features=False,
    )
    return SimpleLinearAttentionPFN(backbone).to(device=device, dtype=DTYPE)


MODEL_CONFIGS = [
    # {
    #     'name': 'non_causal',
    #     'causal_train_only': False,
    #     'display_name': 'Non-causal train context',
    # },
    {
        'name': 'causal_train_only',
        'causal_train_only': True,
        'display_name': 'Causal train context',
    },
]


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    gen_device = device.type if device.type == 'cuda' else 'cpu'
    return torch.Generator(device=gen_device).manual_seed(seed)


def forward_loss_from_context_and_queries(
    model: SimpleLinearAttentionPFN,
    *,
    context_x: torch.Tensor,
    context_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=USE_BF16):
        logits = model(context_x=context_x, context_y=context_y, query_x=query_x)
    targets = query_y.long()
    loss = F.cross_entropy(logits.float().reshape(-1, 2), targets.reshape(-1))
    return loss, logits


def sample_train_context_len(generator: torch.Generator) -> int:
    return int(
        torch.randint(
            low=64,
            high=MAX_TRAIN_CONTEXT_LEN + 1,
            size=(),
            device=DEVICE,
            generator=generator,
        )
    )


def pretrain_model(
    name: str,
    *,
    causal_train_only: bool,
    device: torch.device,
) -> tuple[SimpleLinearAttentionPFN, list[dict[str, float]]]:
    seed_everything(SEED)
    model = make_linear_attention_model(
        causal_train_only=causal_train_only,
        device=device,
    )
    train_model = torch.compile(model) if COMPILE_MODEL else model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    history: list[dict[str, float]] = []
    start = time.time()

    progress = tqdm(range(1, TRAIN_STEPS + 1), desc=f'train {name}')
    for step in progress:
        model.train()
        generator = make_generator(SEED + step, device)
        context_len = sample_train_context_len(generator)
        seq_len = context_len + TEST_LEN

        task = sample_latent_tasks(BATCH_SIZE, device, generator=generator)
        x, y = sample_examples_for_tasks(task, seq_len, generator=generator)

        optimizer.zero_grad(set_to_none=True)
        loss, logits = forward_loss_from_context_and_queries(
            train_model,
            context_x=x[:, :context_len],
            context_y=y[:, :context_len],
            query_x=x[:, context_len:],
            query_y=y[:, context_len:],
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        if LOG_EVERY and (step == 1 or step % LOG_EVERY == 0 or step == TRAIN_STEPS):
            with torch.no_grad():
                targets = y[:, context_len:].long()
                accuracy = (logits.argmax(dim=-1) == targets).float().mean().item()
            row = {
                'step': float(step),
                'context_len': float(context_len),
                'single_eval_pos': float(context_len),
                'loss': float(loss.detach().cpu()),
                'accuracy': accuracy,
                'elapsed_sec': time.time() - start,
            }
            history.append(row)
            if hasattr(progress, 'set_postfix'):
                progress.set_postfix(loss=f"{row['loss']:.3f}", acc=f'{accuracy:.3f}')
    model.eval()
    return model, history

def experiment_signature() -> dict[str, Any]:
    return {
        'seed': SEED,
        'device': str(DEVICE),
        'dtype': str(DTYPE).replace('torch.', ''),
        'num_features': NUM_FEATURES,
        'max_train_context_len': MAX_TRAIN_CONTEXT_LEN,
        'test_len': TEST_LEN,
        'train_steps': TRAIN_STEPS,
        'batch_size': BATCH_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'num_heads': NUM_HEADS,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'grad_clip_norm': GRAD_CLIP_NORM,
        'compile_model': COMPILE_MODEL,
        'log_every': LOG_EVERY,
        'prior_mlp_hidden_size': PRIOR_MLP_HIDDEN_SIZE,
        'prior_mlp_max_hidden_layers': PRIOR_MLP_MAX_HIDDEN_LAYERS,
        'prior_activations': tuple(PRIOR_ACTIVATIONS),
        'eval_context_lengths': tuple(EVAL_CONTEXT_LENGTHS),
        'eval_batch_size': EVAL_BATCH_SIZE,
        'eval_batches': EVAL_BATCHES,
        'models': [dict(model_cfg) for model_cfg in MODEL_CONFIGS],
    }


def checkpoint_args(name: str, causal_train_only: bool) -> dict[str, Any]:
    return {
        **experiment_signature(),
        'name': name,
        'causal_train_only': causal_train_only,
    }


def model_checkpoint_path_from_args(args: dict[str, Any]) -> Path:
    key = hashlib.sha1(json.dumps(args, sort_keys=True).encode()).hexdigest()[:10]
    return CHECKPOINT_DIR / f"{args['name']}_{key}.pt"


def model_checkpoint_path(name: str, causal_train_only: bool) -> Path:
    return model_checkpoint_path_from_args(checkpoint_args(name, causal_train_only))


def save_trained_model(
    name: str,
    model: SimpleLinearAttentionPFN,
    history: list[dict[str, float]],
    *,
    causal_train_only: bool,
) -> Path:
    args = checkpoint_args(name, causal_train_only)
    path = model_checkpoint_path_from_args(args)
    torch.save(
        {
            'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'history': history,
            'args': args,
        },
        path,
    )
    return path


def load_trained_model(
    name: str,
    *,
    causal_train_only: bool,
    device: torch.device = DEVICE,
) -> tuple[SimpleLinearAttentionPFN, list[dict[str, float]]] | None:
    path = model_checkpoint_path(name, causal_train_only)
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location=device)
    model = make_linear_attention_model(causal_train_only=causal_train_only, device=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f'Loaded {name} from {path}')
    return model, checkpoint.get('history', [])


def train_or_load_model(
    name: str,
    *,
    causal_train_only: bool,
) -> tuple[SimpleLinearAttentionPFN, list[dict[str, float]]]:
    loaded = None if FORCE_RETRAIN else load_trained_model(
        name,
        causal_train_only=causal_train_only,
    )
    if loaded is not None:
        return loaded
    model, history = pretrain_model(
        name,
        causal_train_only=causal_train_only,
        device=DEVICE,
    )
    path = save_trained_model(
        name,
        model,
        history,
        causal_train_only=causal_train_only,
    )
    print(f'Saved {name} to {path}')
    return model, history


@torch.inference_mode()
def evaluate_sequence_lengths(
    models: dict[str, SimpleLinearAttentionPFN],
    *,
    device: torch.device,
) -> pd.DataFrame:
    for model in models.values():
        model.eval()

    rows: list[dict[str, float | int | str]] = []
    max_context_len = max(EVAL_CONTEXT_LENGTHS)
    progress = tqdm(range(EVAL_BATCHES), desc='eval sequence lengths')
    for batch_idx in progress:
        generator = make_generator(SEED + 100_000 + batch_idx, device)
        task = sample_latent_tasks(EVAL_BATCH_SIZE, device, generator=generator)
        context_x, context_y = sample_examples_for_tasks(
            task,
            max_context_len,
            generator=generator,
        )
        query_x, query_y = sample_examples_for_tasks(task, TEST_LEN, generator=generator)
        targets = query_y.long()

        for context_len in EVAL_CONTEXT_LENGTHS:
            prefix_x = context_x[:, :context_len]
            prefix_y = context_y[:, :context_len]
            for model_cfg in MODEL_CONFIGS:
                name = model_cfg['name']
                loss, logits = forward_loss_from_context_and_queries(
                    models[name],
                    context_x=prefix_x,
                    context_y=prefix_y,
                    query_x=query_x,
                    query_y=query_y,
                )
                accuracy = (logits.argmax(dim=-1) == targets).float().mean()
                for metric, value in (
                    ('ce', loss.detach()),
                    ('acc', accuracy.detach()),
                ):
                    rows.append(
                        {
                            'model': name,
                            'display_name': model_cfg['display_name'],
                            'seqlen': int(context_len),
                            'rep': int(batch_idx),
                            'metric': metric,
                            'value': float(value.cpu()),
                        }
                    )
                del loss, logits, accuracy
    return pd.DataFrame(rows)


def plot_sequence_length_results(
    results_df: pd.DataFrame,
    *,
    style_map: dict[str, Any],
):
    return plot_curves_from_df(
        results_df,
        specs=[
            ('ce', 'Cross Entropy'),
            ('acc', 'Accuracy'),
        ],
        style_map=style_map,
        x_col='seqlen',
        x_label='Context length',
        title_suffix=' on IID queries',
        error_bars='ci95',
        error_style='band',
        log_x=True,
        show_pretraining_split=True,
        pretrain_boundary=MAX_TRAIN_CONTEXT_LEN,
        model_legend_layout='bottom',
        figsize=(12, 4.8),
    )

def build_normalized_score_df(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_directions = {
        'ce': 'lower',
        'acc': 'higher',
    }
    score_df = results_df[results_df['metric'].isin(metric_directions)].copy()
    grouped_values = score_df.groupby(['rep', 'metric'], observed=True)['value']
    min_values = grouped_values.transform('min')
    max_values = grouped_values.transform('max')
    denominators = max_values - min_values
    score_df['normalized_metric_score'] = 0.5

    non_tied = denominators != 0
    lower_is_better = score_df['metric'].map(metric_directions) == 'lower'
    score_df.loc[non_tied & lower_is_better, 'normalized_metric_score'] = (
        (max_values - score_df['value']) / denominators
    )[non_tied & lower_is_better]
    score_df.loc[non_tied & ~lower_is_better, 'normalized_metric_score'] = (
        (score_df['value'] - min_values) / denominators
    )[non_tied & ~lower_is_better]
    score_df['value'] = score_df['normalized_metric_score']
    score_df['metric'] = 'normalized_' + score_df['metric'].astype(str)
    return score_df.drop(columns=['normalized_metric_score'])


def plot_normalized_sequence_length_scores(
    results_df: pd.DataFrame,
    *,
    style_map: dict[str, Any],
):
    normalized_df = build_normalized_score_df(results_df)
    return plot_curves_from_df(
        normalized_df,
        specs=[
            ('normalized_ce', 'Cross Entropy'),
            ('normalized_acc', 'Accuracy'),
        ],
        style_map=style_map,
        x_col='seqlen',
        x_label='Context length',
        title_suffix=' across models and sequence lengths',
        error_bars='ci95',
        error_style='band',
        log_x=True,
        show_pretraining_split=True,
        pretrain_boundary=MAX_TRAIN_CONTEXT_LEN,
        model_legend_layout='bottom',
        figsize=(12, 4.8),
    )


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip().replace("_", "")) for item in value.split(",") if item.strip())


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _set_model_configs(model_selection: str) -> None:
    global MODEL_CONFIGS
    configs = {
        'non_causal': {
            'name': 'non_causal',
            'causal_train_only': False,
            'display_name': 'Non-causal train context',
        },
        'causal': {
            'name': 'causal_train_only',
            'causal_train_only': True,
            'display_name': 'Causal train context',
        },
    }
    if model_selection == 'both':
        MODEL_CONFIGS = [configs['non_causal'], configs['causal']]
    else:
        MODEL_CONFIGS = [configs[model_selection]]


def _make_run_dir(output_root: Path, run_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or f"run_seed{SEED}_{timestamp}"
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _log_run_to_wandb(
    *,
    run_dir: Path,
    run_name: str,
    config: dict[str, Any],
    project: str,
    entity: str | None,
) -> None:
    import wandb

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config,
    )
    try:
        wandb.save(str(run_dir / 'args.json'), base_path=str(run_dir))
        wandb.log(
            {
                'plots/seq_len_raw': wandb.Image(str(run_dir / 'plots' / 'seq_len_raw.png')),
                'plots/seq_len_normalized': wandb.Image(
                    str(run_dir / 'plots' / 'seq_len_normalized.png')
                ),
            }
        )
    finally:
        run.finish()


def run_experiment(
    run_dir: Path,
    *,
    use_wandb: bool = False,
    wandb_project: str = 'simple_linear_attention_seq_len',
    wandb_entity: str | None = None,
) -> None:
    global SAVE_DIR, CHECKPOINT_DIR

    SAVE_DIR = run_dir
    CHECKPOINT_DIR = SAVE_DIR / 'model_checkpoints'
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = SAVE_DIR / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    config = experiment_signature()
    _write_json(SAVE_DIR / 'args.json', config)
    print(f'device: {DEVICE}, dtype: {DTYPE}')
    print(f'writing run outputs to: {SAVE_DIR}')

    models: dict[str, SimpleLinearAttentionPFN] = {}
    histories: dict[str, list[dict[str, float]]] = {}
    for model_cfg in MODEL_CONFIGS:
        model, history = train_or_load_model(
            model_cfg['name'],
            causal_train_only=model_cfg['causal_train_only'],
        )
        models[model_cfg['name']] = model
        histories[model_cfg['name']] = history

    history_rows = [
        {'model': model_name, **row}
        for model_name, rows in histories.items()
        for row in rows
    ]
    pd.DataFrame(history_rows).to_csv(SAVE_DIR / 'training_history.csv', index=False)

    seq_len_results_df = evaluate_sequence_lengths(models, device=DEVICE)
    seq_len_results_df.to_csv(SAVE_DIR / 'seq_len_results.csv', index=False)

    model_order = [model_cfg['name'] for model_cfg in MODEL_CONFIGS]
    style_map = build_model_style_map(model_order)

    seq_len_fig, _ = plot_sequence_length_results(seq_len_results_df, style_map=style_map)
    seq_len_fig.savefig(plots_dir / 'seq_len_raw.png', dpi=180, bbox_inches='tight')
    plt.close(seq_len_fig)

    normalized_seq_len_scores_df = build_normalized_score_df(seq_len_results_df)
    normalized_seq_len_scores_df.to_csv(SAVE_DIR / 'seq_len_normalized_results.csv', index=False)
    normalized_seq_len_fig, _ = plot_normalized_sequence_length_scores(
        seq_len_results_df,
        style_map=style_map,
    )
    normalized_seq_len_fig.savefig(
        plots_dir / 'seq_len_normalized.png',
        dpi=180,
        bbox_inches='tight',
    )
    plt.close(normalized_seq_len_fig)

    if use_wandb:
        _log_run_to_wandb(
            run_dir=SAVE_DIR,
            run_name=SAVE_DIR.name,
            config=config,
            project=wandb_project,
            entity=wandb_entity,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('simple_linear_attention_runs'))
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--models', choices=('causal', 'non_causal', 'both'), default='causal')
    parser.add_argument('--force-retrain', action='store_true')
    parser.add_argument('--compile-model', action='store_true')
    parser.add_argument('--log-every', type=int, default=LOG_EVERY)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='simple_linear_attention_seq_len')
    parser.add_argument('--wandb-entity', type=str, default=None)

    parser.add_argument('--train-steps', type=int, default=TRAIN_STEPS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--max-train-context-len', type=int, default=MAX_TRAIN_CONTEXT_LEN)
    parser.add_argument('--test-len', type=int, default=TEST_LEN)
    parser.add_argument('--hidden-size', type=int, default=HIDDEN_SIZE)
    parser.add_argument('--num-layers', type=int, default=NUM_LAYERS)
    parser.add_argument('--num-heads', type=int, default=NUM_HEADS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--grad-clip-norm', type=float, default=GRAD_CLIP_NORM)

    parser.add_argument('--prior-hidden-size', type=int, default=PRIOR_MLP_HIDDEN_SIZE)
    parser.add_argument('--prior-max-hidden-layers', type=int, default=PRIOR_MLP_MAX_HIDDEN_LAYERS)
    parser.add_argument('--prior-activations', type=_parse_str_tuple, default=PRIOR_ACTIVATIONS)

    parser.add_argument('--eval-context-lengths', type=_parse_int_tuple, default=EVAL_CONTEXT_LENGTHS)
    parser.add_argument('--eval-batch-size', type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument('--eval-batches', type=int, default=EVAL_BATCHES)
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    global DEVICE, USE_BF16, DTYPE
    global SEED, FORCE_RETRAIN, COMPILE_MODEL, LOG_EVERY
    global TRAIN_STEPS, BATCH_SIZE, MAX_TRAIN_CONTEXT_LEN, TEST_LEN
    global HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS, LR, WEIGHT_DECAY, GRAD_CLIP_NORM
    global PRIOR_MLP_HIDDEN_SIZE, PRIOR_MLP_MAX_HIDDEN_LAYERS, PRIOR_ACTIVATIONS
    global PRIOR_ACTIVATION_MODULES
    global EVAL_CONTEXT_LENGTHS, EVAL_BATCH_SIZE, EVAL_BATCHES

    DEVICE = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device))
    USE_BF16 = DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported()
    DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

    SEED = args.seed
    FORCE_RETRAIN = args.force_retrain
    COMPILE_MODEL = args.compile_model
    LOG_EVERY = max(0, int(args.log_every))

    TRAIN_STEPS = args.train_steps
    BATCH_SIZE = args.batch_size
    MAX_TRAIN_CONTEXT_LEN = args.max_train_context_len
    TEST_LEN = args.test_len
    HIDDEN_SIZE = args.hidden_size
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    GRAD_CLIP_NORM = args.grad_clip_norm

    PRIOR_MLP_HIDDEN_SIZE = args.prior_hidden_size
    PRIOR_MLP_MAX_HIDDEN_LAYERS = args.prior_max_hidden_layers
    PRIOR_ACTIVATIONS = tuple(args.prior_activations)
    unknown_activations = sorted(set(PRIOR_ACTIVATIONS) - set(ACTIVATION_MAP))
    if unknown_activations:
        raise ValueError(f'Unknown prior activations: {unknown_activations}')
    PRIOR_ACTIVATION_MODULES = tuple(ACTIVATION_MAP[name] for name in PRIOR_ACTIVATIONS)

    EVAL_CONTEXT_LENGTHS = tuple(args.eval_context_lengths)
    EVAL_BATCH_SIZE = args.eval_batch_size
    EVAL_BATCHES = args.eval_batches
    _set_model_configs(args.models)


def main() -> None:
    args = parse_args()
    apply_args(args)
    run_dir = _make_run_dir(args.output_root, args.run_name)
    run_experiment(
        run_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )


if __name__ == '__main__':
    main()
