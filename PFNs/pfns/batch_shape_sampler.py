# We create a batch shape sampler, that determines the parts of the batch that determine how compute intensive it is.
# Putting this into one module is important to allow multi-gpu training, as we want to seed the sampling of this shape the same across workers
# s.t. they all take the same amount of time.

# It includes batch_size, seq_len, num_features, and single_eval_pos.

import random
from dataclasses import dataclass
from typing import Optional, Sequence

from pfns.base_config import BaseConfig


@dataclass
class BatchShape:
    batch_size: int
    seq_len: int
    num_features: int
    single_eval_pos: Optional[int] = None
    optimizer_step_progress: float = 1.0

    def as_get_batch_kwargs(self):
        return {
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "num_features": self.num_features,
            "single_eval_pos": self.single_eval_pos,
        }


@dataclass(frozen=True)
class BatchShapeSamplerConfig(BaseConfig):
    batch_size: int = 32
    min_single_eval_pos: int = 0
    max_seq_len: int = 1000
    min_num_features: int = 1
    max_num_features: int = 16
    fixed_num_test_instances: Optional[int] = None
    seq_len_choices: Optional[Sequence[int]] = None
    seq_len_choice_weights: Optional[Sequence[float]] = None
    uniform_seq_len_min: Optional[int] = None
    seq_len_curriculum_start: Optional[int] = None
    seq_len_curriculum_warmup_epochs: int = 0
    seq_len_choice_weight_exponent: float | None = None
    dynamic_batch_size_power: int = 0
    dynamic_batch_size_compensate_grad_accumulation: bool = False

    seed: int = 42

    def __post_init__(self):
        super().__post_init__()

        max_seq_len = int(self.max_seq_len)
        warmup_epochs = int(self.seq_len_curriculum_warmup_epochs)
        curriculum_start = (
            int(self.seq_len_curriculum_start)
            if self.seq_len_curriculum_start is not None
            else None
        )
        choice_weight_exponent = (
            float(self.seq_len_choice_weight_exponent)
            if self.seq_len_choice_weight_exponent is not None
            else None
        )
        choices = (
            tuple(int(v) for v in self.seq_len_choices)
            if self.seq_len_choices is not None
            else None
        )
        weights = (
            tuple(float(w) for w in self.seq_len_choice_weights)
            if self.seq_len_choice_weights is not None
            else None
        )
        uniform_seq_len_min = (
            int(self.uniform_seq_len_min)
            if self.uniform_seq_len_min is not None
            else None
        )
        dynamic_batch_size_power = int(self.dynamic_batch_size_power)
        dynamic_batch_size_compensate_grad_accumulation = bool(
            self.dynamic_batch_size_compensate_grad_accumulation
        )

        assert max_seq_len >= 2, "max_seq_len must be >= 2."
        assert self.batch_size >= 1, "batch_size must be >= 1."
        assert self.min_single_eval_pos >= 0, "min_single_eval_pos must be >= 0."
        assert (
            self.fixed_num_test_instances is None or self.fixed_num_test_instances >= 0
        ), "fixed_num_test_instances must be >= 0 when set."
        assert warmup_epochs >= 0, "seq_len_curriculum_warmup_epochs must be >= 0."

        assert choices is not None or weights is None, (
            "seq_len_choice_weights requires seq_len_choices to be set."
        )
        assert not (choices is not None and uniform_seq_len_min is not None), (
            "uniform_seq_len_min cannot be used together with seq_len_choices."
        )
        if choices is not None:
            assert all(seq_len >= 2 for seq_len in choices) and len(choices) > 0, (
                "All seq_len_choices values must be >= 2 and there must be at least one choice." 
            )
        if uniform_seq_len_min is not None:
            assert uniform_seq_len_min >= 2, "uniform_seq_len_min must be >= 2 when set."
            assert uniform_seq_len_min <= max_seq_len, (
                "uniform_seq_len_min must be <= max_seq_len. "
                f"Got uniform_seq_len_min={uniform_seq_len_min}, max_seq_len={max_seq_len}."
            )
        if weights is not None:
            assert len(weights) == len(choices) and all(weight >= 0.0 for weight in weights), (
                "seq_len_choice_weights must be >= 0 and have the same length as seq_len_choices."
            )

        if curriculum_start is not None:
            assert curriculum_start >= 2, "seq_len_curriculum_start must be >= 2 when set."
        if dynamic_batch_size_power not in (0, 1, 2):
            raise ValueError(
                "dynamic_batch_size_power must be one of {0, 1, 2}. "
                "Use 1 for linear-attention-like scaling and 2 for transformer-like scaling."
            )

        for field_name, value in (
            ("max_seq_len", max_seq_len),
            ("seq_len_choices", choices),
            ("seq_len_choice_weights", weights),
            ("uniform_seq_len_min", uniform_seq_len_min),
            ("seq_len_curriculum_start", curriculum_start),
            ("seq_len_curriculum_warmup_epochs", warmup_epochs),
            ("seq_len_choice_weight_exponent", choice_weight_exponent),
            ("dynamic_batch_size_power", dynamic_batch_size_power),
            (
                "dynamic_batch_size_compensate_grad_accumulation",
                dynamic_batch_size_compensate_grad_accumulation,
            ),
        ):
            object.__setattr__(self, field_name, value)

    def _curriculum_progress(self, epoch: int) -> float:
        warmup = self.seq_len_curriculum_warmup_epochs
        if warmup <= 0:
            return 1.0
        return min(max(float(epoch - 1), 0.0) / float(warmup), 1.0)

    def _effective_max_seq_len(self, epoch: int) -> int:
        """Determine the effective maximum sequence length for the current epoch based on curriculum learning settings."""
        start = self.seq_len_curriculum_start
        warmup = self.seq_len_curriculum_warmup_epochs
        if start is None or warmup <= 0 or self.max_seq_len <= start:
            return self.max_seq_len
        progress = self._curriculum_progress(epoch)
        return int(round(start + progress * (self.max_seq_len - start)))

    def _sample_seq_len_cap(
        self, rng: random.Random, *, max_seq_len_cap: int, epoch: int
    ) -> int:
        """ """
        if not self.seq_len_choices:
            if self.uniform_seq_len_min is not None:
                if self.uniform_seq_len_min > max_seq_len_cap:
                    raise ValueError(
                        "uniform_seq_len_min exceeds the current max_seq_len_cap. "
                        f"Got uniform_seq_len_min={self.uniform_seq_len_min}, "
                        f"max_seq_len_cap={max_seq_len_cap}."
                    )
                return rng.randint(self.uniform_seq_len_min, max_seq_len_cap)
            return max_seq_len_cap
        progress = self._curriculum_progress(epoch)
        source_weights = self.seq_len_choice_weights
        if source_weights is None:
            source_weights = [1.0] * len(self.seq_len_choices)

        choices, base_weights = [], []
        for value, weight in zip(self.seq_len_choices, source_weights):
            if value <= max_seq_len_cap:
                choices.append(value)
                base_weights.append(weight)
        if not choices:
            raise ValueError(
                "No valid seq_len_choices found under max_seq_len_cap. "
                f"max_seq_len_cap={max_seq_len_cap}, seq_len_choices={list(self.seq_len_choices)}."
            )

        if self.seq_len_choice_weight_exponent is not None:
            exponent = progress * self.seq_len_choice_weight_exponent
            min_choice = float(min(choices))
            weights = [
                base_weight * ((float(value) / min_choice) ** exponent)
                for value, base_weight in zip(choices, base_weights)
            ]
        else:
            weights = base_weights

        return (
            rng.choice(choices)
            if sum(weights) <= 0.0
            else rng.choices(choices, weights=weights, k=1)[0]
        )

    def _dynamic_batch_size(self, seq_len: int) -> int:
        if self.dynamic_batch_size_power <= 0:
            return self.batch_size

        # Calibrate memory budget at the smallest sequence length the sampler can emit.
        # This ensures batch size shrinks as seq_len grows in curriculum/uniform settings.
        reference_seq_len = self.max_seq_len
        if self.seq_len_curriculum_start is not None:
            reference_seq_len = self.seq_len_curriculum_start
        elif self.uniform_seq_len_min is not None:
            reference_seq_len = self.uniform_seq_len_min
        elif self.seq_len_choices is not None:
            reference_seq_len = min(self.seq_len_choices)

        # Keep nominal memory at the reference sequence length.
        memory_budget = float(self.batch_size) * (
            float(reference_seq_len) ** self.dynamic_batch_size_power
        )
        raw_batch_size = int(
            memory_budget / (float(seq_len) ** self.dynamic_batch_size_power)
        )
        return max(1, min(self.batch_size, raw_batch_size))

    def _optimizer_step_progress(self, dynamic_batch_size: int) -> float:
        if not self.dynamic_batch_size_compensate_grad_accumulation:
            return 1.0
        return float(dynamic_batch_size) / float(self.batch_size)

    def sample_batch_shape(self, epoch: int, step: int) -> BatchShape:
        # Create deterministic seed based on epoch and step
        seed = self.seed + epoch * 10000 + step
        rng = random.Random(seed)

        # it seems to be beneficial to oversample small numbers of features
        num_features = rng.randint(self.min_num_features, self.max_num_features)

        max_seq_len_cap = self._effective_max_seq_len(epoch)
        seq_len_cap = self._sample_seq_len_cap(
            rng, max_seq_len_cap=max_seq_len_cap, epoch=epoch
        )
        # print(
        #     f"Epoch {epoch}, Step {step}: Sampled seq_len_cap={seq_len_cap} (effective_max_seq_len={max_seq_len_cap}) based on curriculum progress."
        # )
        max_single_eval_pos = (
            seq_len_cap
            - 1
            - (
                self.fixed_num_test_instances
                if self.fixed_num_test_instances is not None
                else 0
            )
        )
        if max_single_eval_pos < 0:
            raise ValueError(
                "Sampled seq_len is too small for fixed_num_test_instances. "
                f"Got seq_len={seq_len_cap}, fixed_num_test_instances={self.fixed_num_test_instances}."
            )
        configured_min_single_eval_pos = int(self.min_single_eval_pos)
        if configured_min_single_eval_pos > max_single_eval_pos:
            raise ValueError(
                "Configured min_single_eval_pos exceeds the maximum allowed single_eval_pos. "
                f"Got min_single_eval_pos={configured_min_single_eval_pos}, "
                f"max_single_eval_pos={max_single_eval_pos}, seq_len_cap={seq_len_cap}, "
                f"fixed_num_test_instances={self.fixed_num_test_instances}."
            )
        min_single_eval_pos = configured_min_single_eval_pos
        single_eval_pos = rng.randint(min_single_eval_pos, max_single_eval_pos)

        seq_len = seq_len_cap
        if self.fixed_num_test_instances is not None:
            seq_len = self.fixed_num_test_instances + single_eval_pos

        dynamic_batch_size = self._dynamic_batch_size(seq_len)
        # print(
        #     f"Epoch {epoch}, Step {step}: Sampled batch shape - batch_size={dynamic_batch_size}, "
        #     f"seq_len={seq_len}, num_features={num_features}, single_eval_pos={single_eval_pos}, "
        #     f"optimizer_step_progress={self._optimizer_step_progress(dynamic_batch_size):.4f}"
        # )
        return BatchShape(
            batch_size=dynamic_batch_size,
            seq_len=seq_len,
            num_features=num_features,
            single_eval_pos=single_eval_pos,
            optimizer_step_progress=self._optimizer_step_progress(dynamic_batch_size),
        )
