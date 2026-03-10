# We create a batch shape sampler, that determines the parts of the batch that determine how compute intensive it is.
# Putting this into one module is important to allow multi-gpu training, as we want to seed the sampling of this shape the same across workers
# s.t. they all take the same amount of time.

# It includes batch_size, seq_len, num_features, and single_eval_pos.

import math
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
    batch_size_stages: Optional[Sequence[tuple[int, int]]] = None
    dynamic_batch_size_compensate_grad_accumulation: bool = False
    min_num_features: int = 1
    max_num_features: int = 20
    fixed_num_test_instances: Optional[int] = None
    eval_pos_split_pct_min: float | None = None
    eval_pos_split_pct_max: float | None = None
    seq_len_stages: Optional[
        Sequence[
            tuple[int, int]
            | tuple[int, int, float, float]
            | tuple[int, int, int, str]
            | tuple[int, int, int, str, float, float]
        ]
    ] = None

    seed: int = 42

    def __post_init__(self):
        super().__post_init__()

        max_seq_len = int(self.max_seq_len)
        batch_size = int(self.batch_size)
        min_single_eval_pos = int(self.min_single_eval_pos)
        min_num_features = int(self.min_num_features)
        max_num_features = int(self.max_num_features)
        dynamic_batch_size_compensate_grad_accumulation = bool(
            self.dynamic_batch_size_compensate_grad_accumulation
        )
        fixed_num_test_instances = (
            None
            if self.fixed_num_test_instances is None
            else int(self.fixed_num_test_instances)
        )
        seed = int(self.seed)
        eval_pos_split_pct_min, eval_pos_split_pct_max = self._normalize_eval_pos_pct_range(
            self.eval_pos_split_pct_min,
            self.eval_pos_split_pct_max,
            source="global eval_pos_split_pct range",
        )

        assert max_seq_len >= 2, "max_seq_len must be >= 2."
        assert batch_size >= 1, "batch_size must be >= 1."
        assert min_single_eval_pos >= 0, "min_single_eval_pos must be >= 0."
        assert min_num_features >= 1, "min_num_features must be >= 1."
        assert max_num_features >= min_num_features, (
            "max_num_features must be >= min_num_features."
        )
        assert (
            fixed_num_test_instances is None or fixed_num_test_instances >= 0
        ), "fixed_num_test_instances must be >= 0 when set."

        resolved_batch_size_stages = self._resolve_batch_size_stages()
        resolved_seq_len_stages = self._resolve_seq_len_stages(max_seq_len)

        for field_name, value in (
            ("batch_size", batch_size),
            ("min_single_eval_pos", min_single_eval_pos),
            ("max_seq_len", max_seq_len),
            ("batch_size_stages", resolved_batch_size_stages),
            (
                "dynamic_batch_size_compensate_grad_accumulation",
                dynamic_batch_size_compensate_grad_accumulation,
            ),
            ("min_num_features", min_num_features),
            ("max_num_features", max_num_features),
            ("fixed_num_test_instances", fixed_num_test_instances),
            ("eval_pos_split_pct_min", eval_pos_split_pct_min),
            ("eval_pos_split_pct_max", eval_pos_split_pct_max),
            ("seq_len_stages", resolved_seq_len_stages),
            ("seed", seed),
        ):
            object.__setattr__(self, field_name, value)

    def _resolve_batch_size_stages(self) -> tuple[tuple[int, int], ...] | None:
        """
        Converts sequence of tuples of (seq_len_threshold, batch_size) into a 
        validated and sorted tuple of the same, or returns None if not set.
        """
        if self.batch_size_stages is None:
            return None

        parsed_batch_size_stages: list[tuple[int, int]] = []
        for stage_index, stage in enumerate(self.batch_size_stages):
            try:
                stage_seq_len_threshold_raw, stage_batch_size_raw = stage
            except Exception as exc:
                raise ValueError(
                    "Each batch_size_stages entry must be a pair "
                    "(seq_len_threshold, batch_size). "
                    f"Invalid entry at index {stage_index}: {stage!r}"
                ) from exc
            stage_seq_len_threshold = int(stage_seq_len_threshold_raw)
            stage_batch_size = int(stage_batch_size_raw)
            if stage_seq_len_threshold < 2:
                raise ValueError(
                    "batch_size_stages seq_len_threshold must be >= 2. "
                    f"Got {stage_seq_len_threshold} at index {stage_index}."
                )
            if stage_batch_size < 1:
                raise ValueError(
                    "batch_size_stages batch_size must be >= 1. "
                    f"Got {stage_batch_size} at index {stage_index}."
                )
            if (
                parsed_batch_size_stages
                and stage_seq_len_threshold <= parsed_batch_size_stages[-1][0]
            ):
                raise ValueError(
                    "batch_size_stages seq_len_threshold values must be strictly increasing."
                )
            parsed_batch_size_stages.append((stage_seq_len_threshold, stage_batch_size))
        return tuple(parsed_batch_size_stages)

    @staticmethod
    def _normalize_seq_len_distribution(distribution: str) -> str:
        normalized = distribution.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized not in {"uniform", "log_uniform"}:
            raise ValueError(
                "seq_len_distribution must be one of ['uniform', 'log_uniform']."
            )
        return normalized

    def _resolve_seq_len_stages(
        self,
        max_seq_len: int,
    ) -> tuple[tuple[int, int, int, str, float | None, float | None], ...] | None:
        """validates and normalizes seq_len_stages
        """
        if self.seq_len_stages is None:
            return None

        parsed_stages: list[tuple[int, int, int, str, float | None, float | None]] = []
        for stage_index, stage in enumerate(self.seq_len_stages):
            (
                end_epoch,
                stage_min_seq_len,
                stage_max_seq_len,
                stage_seq_len_distribution,
                stage_pct_min,
                stage_pct_max,
            ) = self._parse_seq_len_stage(stage=stage, stage_index=stage_index)

            if end_epoch <= 0:
                raise ValueError(
                    "seq_len_stages end_epoch values must be >= 1. "
                    f"Got {end_epoch} at index {stage_index}."
                )
            if stage_min_seq_len < 2:
                raise ValueError(
                    "seq_len_stages min_seq_len values must be >= 2. "
                    f"Got {stage_min_seq_len} at index {stage_index}."
                )
            if stage_max_seq_len < 2:
                raise ValueError(
                    "seq_len_stages max_seq_len_cap values must be >= 2. "
                    f"Got {stage_max_seq_len} at index {stage_index}."
                )
            if stage_min_seq_len > stage_max_seq_len:
                raise ValueError(
                    "seq_len_stages min_seq_len must be <= max_seq_len_cap. "
                    f"Got {stage_min_seq_len} > {stage_max_seq_len} at index {stage_index}."
                )
            if stage_max_seq_len > max_seq_len:
                raise ValueError(
                    "seq_len_stages max_seq_len_cap values must be <= max_seq_len. "
                    f"Got {stage_max_seq_len} > {max_seq_len} at index {stage_index}."
                )
            if parsed_stages and end_epoch <= parsed_stages[-1][0]:
                raise ValueError(
                    "seq_len_stages end_epoch values must be strictly increasing."
                )
            parsed_stages.append(
                (
                    end_epoch,
                    stage_min_seq_len,
                    stage_max_seq_len,
                    stage_seq_len_distribution,
                    stage_pct_min,
                    stage_pct_max,
                )
            )
        return tuple(parsed_stages)

    def _parse_seq_len_stage(
        self,
        *,
        stage: (
            tuple[int, int]
            | tuple[int, int, float, float]
            | tuple[int, int, int, str]
            | tuple[int, int, int, str, float, float]
        ),
        stage_index: int,
    ) -> tuple[int, int, int, str, float | None, float | None]:
        stage_len = len(stage)
        if stage_len == 2:
            end_epoch_raw, stage_max_seq_len_raw = stage
            stage_min_seq_len_raw = stage_max_seq_len_raw
            stage_seq_len_distribution = "fixed"
            stage_pct_min_raw, stage_pct_max_raw = None, None
        elif stage_len == 4:
            if isinstance(stage[3], str):
                (
                    end_epoch_raw,
                    stage_min_seq_len_raw,
                    stage_max_seq_len_raw,
                    stage_seq_len_distribution_raw,
                ) = stage
                stage_seq_len_distribution = self._normalize_seq_len_distribution(
                    stage_seq_len_distribution_raw
                )
                stage_pct_min_raw, stage_pct_max_raw = None, None
            else:
                (
                    end_epoch_raw,
                    stage_max_seq_len_raw,
                    stage_pct_min_raw,
                    stage_pct_max_raw,
                ) = stage
                stage_min_seq_len_raw = stage_max_seq_len_raw
                stage_seq_len_distribution = "fixed"
        elif stage_len == 6:
            (
                end_epoch_raw,
                stage_min_seq_len_raw,
                stage_max_seq_len_raw,
                stage_seq_len_distribution_raw,
                stage_pct_min_raw,
                stage_pct_max_raw,
            ) = stage
            stage_seq_len_distribution = self._normalize_seq_len_distribution(
                stage_seq_len_distribution_raw
            )
        else:
            raise ValueError(
                "Each seq_len_stages entry must be one of: "
                "(end_epoch, max_seq_len), "
                "(end_epoch, max_seq_len, eval_pos_split_pct_min, eval_pos_split_pct_max), "
                "(end_epoch, min_seq_len, max_seq_len, seq_len_distribution), "
                "(end_epoch, min_seq_len, max_seq_len, seq_len_distribution, eval_pos_split_pct_min, eval_pos_split_pct_max). "
                f"Invalid entry at index {stage_index}: {stage!r}"
            )

        end_epoch = int(end_epoch_raw)
        stage_min_seq_len = int(stage_min_seq_len_raw)
        stage_max_seq_len = int(stage_max_seq_len_raw)
        stage_pct_min, stage_pct_max = self._normalize_eval_pos_pct_range(
            stage_pct_min_raw,
            stage_pct_max_raw,
            source=f"seq_len_stages[{stage_index}] eval_pos_split_pct range",
        )
        return (
            end_epoch,
            stage_min_seq_len,
            stage_max_seq_len,
            stage_seq_len_distribution,
            stage_pct_min,
            stage_pct_max,
        )

    @staticmethod
    def _normalize_eval_pos_pct_range(
        pct_min: float | None,
        pct_max: float | None,
        *,
        source: str,
    ) -> tuple[float | None, float | None]:
        if pct_min is None and pct_max is None:
            return None, None
        if pct_min is None:
            pct_min = pct_max
        if pct_max is None:
            pct_max = pct_min

        resolved_min = float(pct_min)
        resolved_max = float(pct_max)
        if resolved_min < 0.0 or resolved_max < 0.0:
            raise ValueError(f"{source} values must be >= 0.")
        if resolved_min > 100.0 or resolved_max > 100.0:
            raise ValueError(f"{source} values must be <= 100.")
        if resolved_min > resolved_max:
            raise ValueError(
                f"{source} min value must be <= max value; got {resolved_min} > {resolved_max}."
            )
        return resolved_min, resolved_max

    def _effective_stage_settings(
        self, epoch: int
    ) -> tuple[int, int, str, float | None, float | None]:
        if not self.seq_len_stages:
            return (
                self.max_seq_len,
                self.max_seq_len,
                "fixed",
                self.eval_pos_split_pct_min,
                self.eval_pos_split_pct_max,
            )
        for (
            end_epoch,
            stage_min_seq_len,
            stage_max_seq_len,
            stage_seq_len_distribution,
            stage_eval_pct_min,
            stage_eval_pct_max,
        ) in self.seq_len_stages:
            if epoch <= end_epoch:
                if stage_eval_pct_min is not None or stage_eval_pct_max is not None:
                    return (
                        stage_min_seq_len,
                        stage_max_seq_len,
                        stage_seq_len_distribution,
                        stage_eval_pct_min,
                        stage_eval_pct_max,
                    )
                return (
                    stage_min_seq_len,
                    stage_max_seq_len,
                    stage_seq_len_distribution,
                    self.eval_pos_split_pct_min,
                    self.eval_pos_split_pct_max,
                )
        return (
            self.max_seq_len,
            self.max_seq_len,
            "fixed",
            self.eval_pos_split_pct_min,
            self.eval_pos_split_pct_max,
        )

    @staticmethod
    def _sample_seq_len(
        *,
        rng: random.Random,
        min_seq_len: int,
        max_seq_len: int,
        seq_len_distribution: str,
    ) -> int:
        if min_seq_len == max_seq_len or seq_len_distribution == "fixed":
            return max_seq_len
        if seq_len_distribution == "uniform":
            return rng.randint(min_seq_len, max_seq_len)
        if seq_len_distribution == "log_uniform":
            sampled = int(
                round(
                    math.exp(
                        rng.uniform(math.log(float(min_seq_len)), math.log(float(max_seq_len)))
                    )
                )
            )
            return max(min_seq_len, min(max_seq_len, sampled))
        raise ValueError(
            f"Unsupported seq_len_distribution {seq_len_distribution!r}; this should have been validated earlier."
        )

    def _dynamic_batch_size(self, seq_len: int) -> int:
        if not self.batch_size_stages:
            return self.batch_size
        for seq_len_threshold, stage_batch_size in self.batch_size_stages:
            if seq_len <= seq_len_threshold:
                return stage_batch_size
        # If seq_len exceeds the largest threshold, keep the smallest configured stage batch size.
        return self.batch_size_stages[-1][1]

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

        (
            stage_min_seq_len,
            stage_max_seq_len,
            stage_seq_len_distribution,
            eval_pct_min,
            eval_pct_max,
        ) = self._effective_stage_settings(epoch)
        seq_len_cap = self._sample_seq_len(
            rng=rng,
            min_seq_len=stage_min_seq_len,
            max_seq_len=stage_max_seq_len,
            seq_len_distribution=stage_seq_len_distribution,
        )
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

        min_single_eval_pos = self.min_single_eval_pos
        if eval_pct_min is not None and eval_pct_max is not None:
            eval_min = int(math.ceil(float(seq_len_cap) * (eval_pct_min / 100.0)))
            eval_max = int(math.floor(float(seq_len_cap) * (eval_pct_max / 100.0)))
            min_single_eval_pos = max(min_single_eval_pos, eval_min)
            max_single_eval_pos = min(max_single_eval_pos, eval_max)

        if min_single_eval_pos > max_single_eval_pos:
            raise ValueError(
                "Configured eval split bounds exceed the maximum allowed "
                "single_eval_pos for the sampled sequence length. "
                f"Got min_single_eval_pos={self.min_single_eval_pos}, "
                f"effective_min_single_eval_pos={min_single_eval_pos}, "
                f"max_single_eval_pos={max_single_eval_pos}, seq_len_cap={seq_len_cap}, "
                f"eval_pos_split_pct_min={eval_pct_min}, "
                f"eval_pos_split_pct_max={eval_pct_max}, "
                f"fixed_num_test_instances={self.fixed_num_test_instances}."
            )
        single_eval_pos = rng.randint(min_single_eval_pos, max_single_eval_pos)

        seq_len = seq_len_cap
        if self.fixed_num_test_instances is not None:
            seq_len = self.fixed_num_test_instances + single_eval_pos

        dynamic_batch_size = self._dynamic_batch_size(seq_len)
        return BatchShape(
            batch_size=dynamic_batch_size,
            seq_len=seq_len,
            num_features=num_features,
            single_eval_pos=single_eval_pos,
            optimizer_step_progress=self._optimizer_step_progress(dynamic_batch_size),
        )
