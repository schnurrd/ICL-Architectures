import pytest

from pfns.batch_shape_sampler import BatchShapeSamplerConfig


def test_seq_len_curriculum_progression_linear_warmup():
    cfg = BatchShapeSamplerConfig(
        batch_size=8,
        max_seq_len=50,
        min_num_features=3,
        max_num_features=3,
        seq_len_curriculum_start=10,
        seq_len_curriculum_warmup_epochs=4,
        seed=7,
    )

    expected_seq_len_by_epoch = {
        1: 10,  # progress = 0.0
        2: 20,  # progress = 0.25
        3: 30,  # progress = 0.5
        4: 40,  # progress = 0.75
        5: 50,  # progress = 1.0
        6: 50,  # saturated at max_seq_len
    }

    for epoch, expected_seq_len in expected_seq_len_by_epoch.items():
        shape = cfg.sample_batch_shape(epoch=epoch, step=0)
        assert shape.seq_len == expected_seq_len
        assert shape.num_features == 3


def test_seq_len_choices_are_filtered_by_curriculum_cap():
    cfg = BatchShapeSamplerConfig(
        batch_size=4,
        max_seq_len=100,
        min_num_features=2,
        max_num_features=2,
        seq_len_choices=[16, 32, 64, 128],
        seq_len_curriculum_start=20,
        seq_len_curriculum_warmup_epochs=2,
        seed=13,
    )

    # Epoch 1 cap=20 -> only 16 is valid.
    epoch1_seq_lens = {
        cfg.sample_batch_shape(epoch=1, step=step).seq_len for step in range(10)
    }
    assert epoch1_seq_lens == {16}

    # Epoch 2 cap=60 -> 16/32 are valid, 64/128 are not.
    epoch2_seq_lens = {
        cfg.sample_batch_shape(epoch=2, step=step).seq_len for step in range(40)
    }
    assert epoch2_seq_lens.issubset({16, 32})
    assert epoch2_seq_lens

    # Epoch 3+ cap=100 -> 16/32/64 are valid, 128 is not.
    epoch3_seq_lens = {
        cfg.sample_batch_shape(epoch=3, step=step).seq_len for step in range(80)
    }
    assert epoch3_seq_lens.issubset({16, 32, 64})
    assert 128 not in epoch3_seq_lens


def test_weighted_seq_len_choices_respected():
    cfg = BatchShapeSamplerConfig(
        batch_size=4,
        max_seq_len=40,
        seq_len_choices=[10, 20, 30],
        seq_len_choice_weights=[0.0, 1.0, 0.0],
        seed=23,
    )

    seq_lens = [cfg.sample_batch_shape(epoch=1, step=step).seq_len for step in range(50)]
    assert set(seq_lens) == {20}


def test_sampling_is_deterministic_for_same_epoch_and_step():
    cfg_a = BatchShapeSamplerConfig(
        batch_size=16,
        max_seq_len=128,
        min_num_features=2,
        max_num_features=10,
        seq_len_choices=[32, 64, 96, 128],
        seq_len_choice_weights=[0.1, 0.2, 0.3, 0.4],
        seq_len_curriculum_start=64,
        seq_len_curriculum_warmup_epochs=3,
        seed=99,
    )
    cfg_b = BatchShapeSamplerConfig(
        batch_size=16,
        max_seq_len=128,
        min_num_features=2,
        max_num_features=10,
        seq_len_choices=[32, 64, 96, 128],
        seq_len_choice_weights=[0.1, 0.2, 0.3, 0.4],
        seq_len_curriculum_start=64,
        seq_len_curriculum_warmup_epochs=3,
        seed=99,
    )

    for epoch in (1, 2, 3, 4):
        for step in (0, 1, 7, 33):
            assert cfg_a.sample_batch_shape(epoch=epoch, step=step) == cfg_b.sample_batch_shape(
                epoch=epoch, step=step
            )


def test_raises_if_sampled_seq_len_is_too_small_for_fixed_test_instances():
    cfg = BatchShapeSamplerConfig(
        batch_size=8,
        max_seq_len=40,
        fixed_num_test_instances=30,
        seq_len_choices=[20],
        seed=5,
    )

    with pytest.raises(
        ValueError,
        match="Sampled seq_len is too small for fixed_num_test_instances",
    ):
        cfg.sample_batch_shape(epoch=1, step=0)


def test_epoch_dependent_choice_weighting_shifts_to_longer_sequences():
    cfg = BatchShapeSamplerConfig(
        batch_size=8,
        max_seq_len=256,
        seq_len_choices=[64, 128, 256],
        seq_len_curriculum_start=64,
        seq_len_curriculum_warmup_epochs=8,
        seq_len_choice_weight_exponent=3.0,
        seed=11,
    )

    early = [cfg.sample_batch_shape(epoch=1, step=step).seq_len for step in range(200)]
    late = [cfg.sample_batch_shape(epoch=10, step=step).seq_len for step in range(200)]

    assert sum(late) / len(late) > sum(early) / len(early)


def test_dynamic_batch_size_scales_with_seq_len_linear_and_quadratic():
    linear_cfg = BatchShapeSamplerConfig(
        batch_size=64,
        max_seq_len=256,
        seq_len_choices=[64, 128, 256],
        dynamic_batch_size_power=1,
    )
    quadratic_cfg = BatchShapeSamplerConfig(
        batch_size=64,
        max_seq_len=256,
        seq_len_choices=[64, 128, 256],
        dynamic_batch_size_power=2,
    )

    assert linear_cfg._dynamic_batch_size(64) == 64
    assert linear_cfg._dynamic_batch_size(128) == 32
    assert linear_cfg._dynamic_batch_size(256) == 16

    assert quadratic_cfg._dynamic_batch_size(64) == 64
    assert quadratic_cfg._dynamic_batch_size(128) == 16
    assert quadratic_cfg._dynamic_batch_size(256) == 4


def test_dynamic_batch_size_uses_uniform_seq_len_min_as_reference():
    cfg = BatchShapeSamplerConfig(
        batch_size=8,
        max_seq_len=160_000,
        uniform_seq_len_min=1_000,
        dynamic_batch_size_power=1,
    )

    assert cfg._dynamic_batch_size(1_000) == 8
    assert cfg._dynamic_batch_size(2_000) == 4
    assert cfg._dynamic_batch_size(4_000) == 2
    assert cfg._dynamic_batch_size(8_000) == 1
    assert cfg._dynamic_batch_size(160_000) == 1


def test_optimizer_step_progress_reflects_dynamic_batch_when_enabled():
    cfg = BatchShapeSamplerConfig(
        batch_size=64,
        max_seq_len=256,
        seq_len_choices=[64, 256],
        seq_len_choice_weights=[0.0, 1.0],
        dynamic_batch_size_power=1,
        dynamic_batch_size_compensate_grad_accumulation=True,
        seed=17,
    )

    shape = cfg.sample_batch_shape(epoch=1, step=0)
    assert shape.seq_len == 256
    assert shape.batch_size == 16
    assert shape.optimizer_step_progress == 0.25


def test_uniform_seq_len_sampling_uses_integer_range_when_enabled():
    cfg = BatchShapeSamplerConfig(
        batch_size=4,
        max_seq_len=20,
        uniform_seq_len_min=10,
        seed=3,
    )
    sampled = [cfg.sample_batch_shape(epoch=1, step=step).seq_len for step in range(100)]
    assert all(10 <= value <= 20 for value in sampled)
    assert len(set(sampled)) > 1


def test_uniform_seq_len_sampling_cannot_be_combined_with_choices():
    with pytest.raises(
        AssertionError,
        match="uniform_seq_len_min cannot be used together with seq_len_choices",
    ):
        BatchShapeSamplerConfig(
            batch_size=4,
            max_seq_len=20,
            seq_len_choices=[10, 20],
            uniform_seq_len_min=10,
        )
