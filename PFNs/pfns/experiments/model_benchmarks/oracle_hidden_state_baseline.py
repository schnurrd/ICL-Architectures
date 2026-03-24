from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn

from pfns.model.backbones import FLABackbone
from pfns.model.tabular_model import InContextState, TabularModel
from pfns.training_utils import compute_losses


@dataclass(frozen=True)
class OracleHiddenStateConfig:
    num_epochs: int = 1
    lr: float = 5e-2
    weight_decay: float = 0.0
    
    # Early stopping parameters
    patience: int = 20 
    tolerance: float = 1e-5 # minimum improvement to reset patience
    selection_fraction: float = 0.0 # for val
    selection_seed: int = 42
    
    query_batch_size: int = 256
    
    # Logging
    verbose: bool = False
    log_every_steps: int = 50

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "OracleHiddenStateConfig":
        return cls(
            num_epochs=int(config.get("oracle_num_epochs", 1)),
            lr=float(config.get("oracle_lr", 5e-2)),
            weight_decay=float(config.get("oracle_weight_decay", 0.0)),
            patience=int(config.get("oracle_patience", 20)),
            tolerance=float(config.get("oracle_tolerance", 1e-5)),
            query_batch_size=int(config.get("oracle_query_batch_size", 256)),
            selection_fraction=float(config.get("oracle_selection_fraction", 0.0)),
            selection_seed=int(config.get("oracle_selection_seed", 42)),
            verbose=bool(config.get("oracle_verbose", False)),
            log_every_steps=int(config.get("oracle_log_every_steps", 50)),
        )

    def __post_init__(self) -> None:
        if self.num_epochs < 1:
            raise ValueError("oracle_num_epochs must be >= 1")
        if self.lr <= 0:
            raise ValueError("oracle_lr must be > 0")
        if self.weight_decay < 0:
            raise ValueError("oracle_weight_decay must be >= 0")
        if self.patience < 1:
            raise ValueError("oracle_patience must be >= 1")
        if self.tolerance < 0:
            raise ValueError("oracle_tolerance must be >= 0")
        if self.query_batch_size < 1:
            raise ValueError("oracle_query_batch_size must be >= 1")
        if not 0.0 <= self.selection_fraction < 1.0:
            raise ValueError("oracle_selection_fraction must be in [0, 1)")
        if self.log_every_steps < 1:
            raise ValueError("oracle_log_every_steps must be >= 1")


class OracleHiddenStateBaseline(nn.Module):
    """Optimize recurrent FLA cache states while keeping the model weights frozen."""

    requires_grad_during_eval = True

    def __init__(
        self,
        *,
        base_model: TabularModel,
        optimization_config: OracleHiddenStateConfig,
    ) -> None:
        super().__init__()
        self.base_model = base_model.eval()
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.criterion = self.base_model.criterion
        self.optimization_config = optimization_config

    def _log(self, message: str) -> None:
        if self.optimization_config.verbose:
            print(f"OracleHiddenStateBaseline: {message}")

    def _extract_recurrent_states(self, cache_params: Any) -> list[nn.Parameter]:
        if not hasattr(cache_params, "layers"):
            raise TypeError("Oracle hidden-state baseline requires an FLA cache with per-layer states.")

        recurrent_states: list[nn.Parameter] = []
        for layer in cache_params.layers:
            state = getattr(layer, "state", None)
            recurrent_state = state.get("recurrent_state") if isinstance(state, dict) else None
            if not torch.is_tensor(recurrent_state):
                raise TypeError(
                    "Oracle hidden-state baseline requires each layer cache to expose "
                    "state['recurrent_state']."
                )
            recurrent_states.append(nn.Parameter(recurrent_state.detach().clone()))
        if not recurrent_states:
            raise ValueError("No recurrent_state tensors were found in the cached FLA state.")
        return recurrent_states

    def _candidate_state(
        self,
        *,
        backbone_state: dict[str, Any],
        cache_params: Any,
        recurrent_states: list[torch.Tensor],
    ) -> dict[str, Any]:
        cache_copy = FLABackbone._shallow_copy(cache_params)
        cache_copy.layers = []
        for layer, recurrent_state in zip(cache_params.layers, recurrent_states, strict=True):
            layer_copy = FLABackbone._shallow_copy(layer)
            state = getattr(layer, "state", None)
            if not isinstance(state, dict):
                raise TypeError("Oracle hidden-state baseline requires dict-like per-layer cache states.")
            layer_copy.state = FLABackbone._copy_state(state)
            layer_copy.state["recurrent_state"] = recurrent_state.clone()
            cache_copy.layers.append(layer_copy)

        state_with_cache = dict(backbone_state)
        state_with_cache["cache_params"] = cache_copy
        return state_with_cache

    def _training_loss(
        self,
        *,
        backbone_state: dict[str, Any],
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        style: torch.Tensor | None,
        y_style: torch.Tensor | None,
        categorical_inds: list[int] | None,
    ) -> torch.Tensor:
        with torch.no_grad():
            query_x_bf, _, _ = self.base_model._prepare_batch_first_inputs(query_x, None, None)
            assert query_x_bf is not None
            embedded_input, current_context_len, should_interleave, int_mt_mode = self.base_model._build_embedded_input(
                query_x_bf,
                None,
                single_eval_pos=None,
                style=style,
                y_style=y_style,
                categorical_inds=categorical_inds,
                cache_trainset_representation=True,
            )
        encoder_out = self.base_model.backbone.incontext_predict(
            embedded_input,
            backbone_state,
            rope_pairwise_positions=should_interleave,
        )
        logits = self.base_model._decode_from_encoder_out(
            encoder_out,
            current_context_len,
            should_interleave,
            int_mt_mode,
        )["standard"]
        losses = compute_losses(
            logits,
            query_y.to(logits.device).clone(),
            self.criterion,
            1,
        )
        return losses.mean()

    def _full_training_loss(
        self,
        *,
        backbone_state: dict[str, Any],
        x: torch.Tensor,
        y: torch.Tensor,
        style: torch.Tensor | None,
        y_style: torch.Tensor | None,
        categorical_inds: list[int] | None,
    ) -> float:
        total_loss = 0.0
        total_weight = 0
        query_batch_size = min(self.optimization_config.query_batch_size, int(x.shape[1]))
        with torch.no_grad():
            for start in range(0, int(x.shape[1]), query_batch_size):
                stop = min(start + query_batch_size, int(x.shape[1]))
                query_x = x[:, start:stop]
                query_y = y[:, start:stop]
                chunk_loss = self._training_loss(
                    backbone_state=backbone_state,
                    query_x=query_x,
                    query_y=query_y,
                    style=style,
                    y_style=y_style,
                    categorical_inds=categorical_inds,
                )
                weight = stop - start
                total_loss += float(chunk_loss.item()) * weight
                total_weight += weight
        if total_weight == 0:
            raise ValueError("Training sequence length must be >= 1.")
        return total_loss / total_weight

    def _sample_query_batch(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        permutation: torch.Tensor,
        perm_offset: int,
        query_batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        train_len = int(x.shape[1])
        remaining = train_len - perm_offset
        if remaining >= query_batch_size:
            query_indices = permutation[perm_offset : perm_offset + query_batch_size]
            next_permutation = permutation
            next_offset = perm_offset + query_batch_size
        else:
            tail = permutation[perm_offset:]
            next_permutation = torch.randperm(train_len, device=x.device, generator=generator)
            needed = query_batch_size - remaining
            head = next_permutation[:needed]
            query_indices = torch.cat((tail, head), dim=0)
            next_offset = needed

        query_x = x.index_select(1, query_indices)
        query_y = y.index_select(1, query_indices)
        return query_x, query_y, next_permutation, next_offset

    def _train_and_val_split(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        train_len = int(x.shape[1])
        if self.optimization_config.selection_fraction <= 0.0 or train_len < 2:
            return x, y, x, y, "train"

        selection_size = max(1, int(round(train_len * self.optimization_config.selection_fraction)))
        selection_size = min(selection_size, train_len - 1)
        generator = torch.Generator(device=x.device)
        generator.manual_seed(self.optimization_config.selection_seed)
        permutation = torch.randperm(train_len, device=x.device, generator=generator)
        selection_indices = permutation[:selection_size]
        optimize_indices = permutation[selection_size:]
        return (
            x.index_select(1, optimize_indices),
            y.index_select(1, optimize_indices),
            x.index_select(1, selection_indices),
            y.index_select(1, selection_indices),
            "val",
        )

    def incontext_fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        style: torch.Tensor | None = None,
        y_style: torch.Tensor | None = None,
        categorical_inds: list[int] | None = None,
    ) -> InContextState:
        with torch.inference_mode(False):
            with torch.no_grad():
                initial_state = self.base_model.incontext_fit(
                    x=x,
                    y=y,
                    style=style,
                    y_style=y_style,
                    categorical_inds=categorical_inds,
                )
            backbone_state = dict(initial_state.backbone_state)
            cache_params = backbone_state.get("cache_params")
            recurrent_states = self._extract_recurrent_states(cache_params)
            optimizer = torch.optim.AdamW(
                recurrent_states,
                lr=self.optimization_config.lr,
                weight_decay=self.optimization_config.weight_decay,
            )
            train_x, train_y, val_x, val_y, selection_name = self._train_and_val_split(x, y)
            train_len = int(train_x.shape[1])
            if train_len < 1:
                raise ValueError("Training sequence length must be >= 1 after train/validation split.")
            query_batch_size = min(self.optimization_config.query_batch_size, train_len)
            steps_per_epoch = math.ceil(train_len / query_batch_size)
            total_steps = steps_per_epoch * self.optimization_config.num_epochs
            optimize_generator = torch.Generator(device=x.device)
            optimize_generator.manual_seed(self.optimization_config.selection_seed + 1)
            common_kwargs = {
                "style": style,
                "y_style": y_style,
                "categorical_inds": categorical_inds,
            }

            def candidate_state() -> dict[str, Any]:
                return self._candidate_state(
                    backbone_state=backbone_state,
                    cache_params=cache_params,
                    recurrent_states=recurrent_states,
                )

            def val_loss() -> float:
                return self._full_training_loss(
                    backbone_state=candidate_state(),
                    x=val_x,
                    y=val_y,
                    **common_kwargs,
                )

            def minibatch_loss(query_x: torch.Tensor, query_y: torch.Tensor) -> torch.Tensor:
                with torch.enable_grad():
                    loss = self._training_loss(
                        backbone_state=candidate_state(),
                        query_x=query_x,
                        query_y=query_y,
                        **common_kwargs,
                    )
                if not loss.requires_grad:
                    raise RuntimeError("Oracle hidden-state optimization produced a detached loss")
                return loss

            best_loss = val_loss()
            best_states = [state.detach().clone() for state in recurrent_states]
            evals_without_improvement = 0
            permutation = torch.randperm(train_len, device=x.device, generator=optimize_generator)
            perm_offset = 0
            self._log(
                f"initial_{selection_name}_loss={best_loss:.6f} "
                f"train_len={train_len} val_len={int(val_x.shape[1])} "
                f"query_batch_size={query_batch_size} total_steps={total_steps}"
            )

            for step_idx in range(total_steps):
                query_x, query_y, permutation, perm_offset = self._sample_query_batch(
                    x=train_x,
                    y=train_y,
                    permutation=permutation,
                    perm_offset=perm_offset,
                    query_batch_size=query_batch_size,
                    generator=optimize_generator,
                )
                loss = minibatch_loss(query_x, query_y)
                loss_value = float(loss.detach().item())

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                should_log_step = step_idx == 0 or (step_idx + 1) % self.optimization_config.log_every_steps == 0
                if should_log_step:
                    self._log(
                        f"step={step_idx + 1}/{total_steps} minibatch_train_loss={loss_value:.6f}"
                    )

                completed_epoch = (step_idx + 1) % steps_per_epoch == 0 or (step_idx + 1) == total_steps
                if not completed_epoch:
                    continue

                full_loss = val_loss()
                epoch_idx = math.ceil((step_idx + 1) / steps_per_epoch)
                self._log(
                    f"epoch={epoch_idx} {selection_name}_loss={full_loss:.6f} "
                    f"best_{selection_name}_loss={best_loss:.6f}"
                )
                if full_loss + self.optimization_config.tolerance < best_loss:
                    best_loss = full_loss
                    best_states = [state.detach().clone() for state in recurrent_states]
                    evals_without_improvement = 0
                    continue

                evals_without_improvement += 1
                if evals_without_improvement >= self.optimization_config.patience:
                    self._log(
                        f"early_stop_after={step_idx + 1} steps best_{selection_name}_loss={best_loss:.6f}"
                    )
                    break

            optimized_state = self._candidate_state(
                backbone_state=backbone_state,
                cache_params=cache_params,
                recurrent_states=best_states,
            )
            return InContextState(backbone_state=optimized_state)

    def incontext_predict(
        self,
        state: InContextState,
        test_x: torch.Tensor,
        *,
        style: torch.Tensor | None = None,
        y_style: torch.Tensor | None = None,
        categorical_inds: list[int] | None = None,
        only_return_standard_out: bool = True,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        return self.base_model.incontext_predict(
            state,
            test_x=test_x,
            style=style,
            y_style=y_style,
            categorical_inds=categorical_inds,
            only_return_standard_out=only_return_standard_out,
        )


def build_oracle_hidden_state_baseline(
    *,
    base_model: TabularModel,
    base_config: Any,
    model_config: Mapping[str, Any],
) -> OracleHiddenStateBaseline:
    return OracleHiddenStateBaseline(
        base_model=base_model,
        optimization_config=OracleHiddenStateConfig.from_mapping(model_config),
    )
