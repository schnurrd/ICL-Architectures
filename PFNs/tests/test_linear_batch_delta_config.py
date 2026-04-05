from configs.transformer.linear_batch_delta_config import get_config


def test_linear_batch_delta_config_default_builds() -> None:
    config = get_config(training_setup="debug")
    backbone = config.model.backbone
    assert backbone.lower_nlayers == 11
    assert backbone.upper_nlayers == 4
    assert backbone.batch_delta_state_dim == 64
    assert backbone.mlp_hidden_dim == 736
    assert backbone.batch_delta_layer_kwargs == {
        "num_solver_steps": 1,
        "support_target_mode": "hidden_plus_label",
        "target_bilinear_rank": 0,
        "fast_weight_rank": 0,
        "base_fast_weight_context_rank": 0,
        "incontext_opt_rank": 0,
        "incontext_opt_steps": 0,
        "incontext_opt_lr": 5e-2,
        "incontext_opt_weight_decay": 0.0,
        "ridge_lambda_init": 1e-1,
        "learnable_ridge_lambda": False,
        "qk_l2_normalize": True,
        "residual_scale_init": 3e-2,
    }

    model = config.model.create_model()
    assert len(model.backbone.lower_layers) == 11
    assert len(model.backbone.upper_layers) == 4


def test_linear_batch_delta_config_roundtrip_is_stable() -> None:
    config = get_config(training_setup="debug")
    assert config.from_yaml(config.to_yaml()) == config
