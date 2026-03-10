import math

import pytest

from pfns.batch_shape_sampler import BatchShapeSamplerConfig


def test_seq_len_stages_progress_by_epoch():
    cfg = BatchShapeSamplerConfig(
        batch_size=8,
        max_seq_len=1000,
        min_num_features=3,
        max_num_features=3,
        seq_len_stages=[(2, 128), (4, 256)],
        seed=7,
    )

    expected_seq_len_by_epoch = {
        1: 128,
        2: 128,
        3: 256,
        4: 256,
        5: 1000,
    }

    for epoch, expected_seq_len in expected_seq_len_by_epoch.items():
        shape = cfg.sample_batch_shape(epoch=epoch, step=0)
        assert shape.seq_len == expected_seq_len
        assert shape.num_features == 3


@pytest.mark.parametrize(
    "pct_min,pct_max,expected_min,expected_max,expect_multiple",
    [
        (80.0, 80.0, 800, 800, False),
        (30.0, 90.0, 300, 900, True),
    ],
)
def test_eval_pos_split_global_modes(
    pct_min: float,
    pct_max: float,
    expected_min: int,
    expected_max: int,
    expect_multiple: bool,
):
    cfg = BatchShapeSamplerConfig(
        batch_size=4,
        max_seq_len=1000,
        eval_pos_split_pct_min=pct_min,
        eval_pos_split_pct_max=pct_max,
        seed=11,
    )

    sampled = [
        cfg.sample_batch_shape(epoch=1, step=step).single_eval_pos
        for step in range(200)
    ]
    assert all(expected_min <= value <= expected_max for value in sampled)
    if expect_multiple:
        assert len(set(sampled)) > 1
    else:
        assert set(sampled) == {expected_min}


def test_stage_eval_pos_split_overrides_global_range():
    cfg = BatchShapeSamplerConfig(
        batch_size=4,
        max_seq_len=500,
        eval_pos_split_pct_min=20.0,
        eval_pos_split_pct_max=40.0,
        seq_len_stages=[(2, 200, 80.0, 80.0), (4, 100, 300, "uniform", 30.0, 35.0)],
        seed=17,
    )

    # Stage 1: fixed seq_len=200 and stage split fixed at 80%.
    assert cfg.sample_batch_shape(epoch=1, step=0).single_eval_pos == 160

    # Stage 2: sampled seq_len in [100, 300] with stage split range 30-35%.
    sampled_stage_2 = [cfg.sample_batch_shape(epoch=3, step=step) for step in range(100)]
    for shape in sampled_stage_2:
        assert 100 <= shape.seq_len <= 300
        assert math.ceil(shape.seq_len * 0.3) <= shape.single_eval_pos <= math.floor(
            shape.seq_len * 0.35
        )

    # Post-stages: fallback to global 20-40% with max_seq_len=500.
    sampled_post_stage = [
        cfg.sample_batch_shape(epoch=5, step=step).single_eval_pos for step in range(100)
    ]
    assert all(100 <= value <= 200 for value in sampled_post_stage)


def test_sampling_is_deterministic_for_same_epoch_and_step():
    cfg_a = BatchShapeSamplerConfig(
        batch_size=16,
        max_seq_len=512,
        min_num_features=2,
        max_num_features=10,
        eval_pos_split_pct_min=60.0,
        eval_pos_split_pct_max=80.0,
        seq_len_stages=[(3, 128), (6, 128, 256, "uniform")],
        seed=99,
    )
    cfg_b = BatchShapeSamplerConfig(
        batch_size=16,
        max_seq_len=512,
        min_num_features=2,
        max_num_features=10,
        eval_pos_split_pct_min=60.0,
        eval_pos_split_pct_max=80.0,
        seq_len_stages=[(3, 128), (6, 128, 256, "uniform")],
        seed=99,
    )

    for epoch in (1, 2, 3, 4, 7):
        for step in (0, 1, 7, 33):
            assert cfg_a.sample_batch_shape(epoch=epoch, step=step) == cfg_b.sample_batch_shape(
                epoch=epoch, step=step
            )


def test_raises_if_sampled_seq_len_is_too_small_for_fixed_test_instances():
    cfg = BatchShapeSamplerConfig(
        batch_size=8,
        max_seq_len=40,
        fixed_num_test_instances=30,
        seq_len_stages=[(10, 20)],
        seed=5,
    )

    with pytest.raises(
        ValueError,
        match="Sampled seq_len is too small for fixed_num_test_instances",
    ):
        cfg.sample_batch_shape(epoch=1, step=0)


@pytest.mark.parametrize(
    "stages,match",
    [
        ([(2, 32, 20.0, 40.0, 60.0, 70.0, 80.0)], "Each seq_len_stages entry must be one of"),
        ([(2, 32, 80.0)], "Each seq_len_stages entry must be one of"),
        ([(2, 16, 32, "normal")], "seq_len_distribution must be one of"),
    ],
)
def test_invalid_seq_len_stage_definitions_raise(stages, match: str):
    with pytest.raises(ValueError, match=match):
        BatchShapeSamplerConfig(
            batch_size=4,
            max_seq_len=64,
            seq_len_stages=stages,
        )


@pytest.mark.parametrize("distribution", ["uniform", "log_uniform"])
def test_sampled_seq_len_stages_distributions(distribution: str):
    cfg = BatchShapeSamplerConfig(
        batch_size=4,
        max_seq_len=1000,
        seq_len_stages=[(5, 100, 300, distribution)],
        seed=13,
    )
    samples = [
        cfg.sample_batch_shape(epoch=1, step=step).seq_len for step in range(100)
    ]
    assert all(100 <= value <= 300 for value in samples)
    assert len(set(samples)) > 1


def test_batch_size_stages_and_compensation_progress():
    cfg = BatchShapeSamplerConfig(
        batch_size=16,
        max_seq_len=1000,
        seq_len_stages=[(2, 120), (4, 400)],
        batch_size_stages=[(128, 16), (512, 8), (2000, 4)],
        seed=3,
    )

    # Stage 1 seq_len=120 -> first threshold batch_size=16
    assert cfg.sample_batch_shape(epoch=1, step=0).batch_size == 16
    # Stage 2 seq_len=400 -> second threshold batch_size=8
    assert cfg.sample_batch_shape(epoch=3, step=0).batch_size == 8
    # Stage 3 seq_len=1000 -> third threshold batch_size=4
    assert cfg.sample_batch_shape(epoch=6, step=0).batch_size == 4

    cfg_compensated = BatchShapeSamplerConfig(
        batch_size=16,
        max_seq_len=1000,
        batch_size_stages=[(256, 16), (1000, 4)],
        dynamic_batch_size_compensate_grad_accumulation=True,
        seed=7,
    )
    shape = cfg_compensated.sample_batch_shape(epoch=1, step=0)
    assert shape.batch_size == 4
    assert shape.optimizer_step_progress == 0.25
