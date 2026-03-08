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
    seq_len_curriculum_start: Optional[int] = None
    seq_len_curriculum_warmup_epochs: int = 0

    seed: int = 42

    def __post_init__(self):
        super().__post_init__()

        max_seq_len = int(self.max_seq_len)
        warmup_epochs = int(self.seq_len_curriculum_warmup_epochs)
        curriculum_start = (
            None
            if self.seq_len_curriculum_start is None
            else int(self.seq_len_curriculum_start)
        )
        choices = (
            None
            if self.seq_len_choices is None
            else tuple(int(v) for v in self.seq_len_choices)
        )
        weights = (
            None
            if self.seq_len_choice_weights is None
            else tuple(float(w) for w in self.seq_len_choice_weights)
        )

        assert max_seq_len >= 2, "max_seq_len must be >= 2."
        assert self.min_single_eval_pos >= 0, "min_single_eval_pos must be >= 0."
        assert (
            self.fixed_num_test_instances is None or self.fixed_num_test_instances >= 0
        ), "fixed_num_test_instances must be >= 0 when set."

        assert choices is not None or weights is None, (
            "seq_len_choice_weights requires seq_len_choices to be set."
        )
        if choices is not None:
            assert all(seq_len >= 2 for seq_len in choices), (
                "All seq_len_choices values must be >= 2."
            )
        if weights is not None:
            assert len(weights) == len(choices), (
                "seq_len_choice_weights must have the same length as seq_len_choices."
            )
            assert all(weight >= 0.0 for weight in weights), (
                "seq_len_choice_weights must be >= 0."
            )

        if curriculum_start is not None:
            assert curriculum_start >= 2, "seq_len_curriculum_start must be >= 2 when set."

        for field_name, value in (
            ("max_seq_len", max_seq_len),
            ("seq_len_choices", choices),
            ("seq_len_choice_weights", weights),
            ("seq_len_curriculum_start", curriculum_start),
            ("seq_len_curriculum_warmup_epochs", warmup_epochs),
        ):
            object.__setattr__(self, field_name, value)

    def _effective_max_seq_len(self, epoch: int) -> int:
        start = self.seq_len_curriculum_start
        warmup = self.seq_len_curriculum_warmup_epochs
        if start is None or warmup <= 0 or self.max_seq_len <= start:
            return self.max_seq_len
        progress = min(max(float(epoch - 1), 0.0) / float(warmup), 1.0)
        return int(round(start + progress * (self.max_seq_len - start)))

    def _sample_seq_len_cap(self, rng: random.Random, *, max_seq_len_cap: int) -> int:
        choices = [value for value in (self.seq_len_choices or ()) if value <= max_seq_len_cap]
        if not choices:
            return max_seq_len_cap
        if self.seq_len_choice_weights is None:
            return rng.choice(choices)

        filtered_weights = [
            weight
            for value, weight in zip(self.seq_len_choices, self.seq_len_choice_weights)
            if value <= max_seq_len_cap
        ]
        return (
            rng.choice(choices)
            if sum(filtered_weights) <= 0.0
            else rng.choices(choices, weights=filtered_weights, k=1)[0]
        )

    def sample_batch_shape(self, epoch: int, step: int) -> BatchShape:
        # Create deterministic seed based on epoch and step
        seed = self.seed + epoch * 10000 + step
        rng = random.Random(seed)

        # it seems to be beneficial to oversample small numbers of features
        num_features = rng.randint(self.min_num_features, self.max_num_features)

        max_seq_len_cap = self._effective_max_seq_len(epoch)
        seq_len_cap = self._sample_seq_len_cap(rng, max_seq_len_cap=max_seq_len_cap)
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
        min_single_eval_pos = min(int(self.min_single_eval_pos), int(max_single_eval_pos))
        single_eval_pos = rng.randint(min_single_eval_pos, max_single_eval_pos)

        seq_len = seq_len_cap
        if self.fixed_num_test_instances is not None:
            seq_len = self.fixed_num_test_instances + single_eval_pos

        # future todo: adapt batch_size and num_features based on seq_len -> shrinking them for large seq_lens
        return BatchShape(
            batch_size=self.batch_size,
            seq_len=seq_len,
            num_features=num_features,
            single_eval_pos=single_eval_pos,
        )
