import pytest
import torch

pytest.importorskip("fla")

from pfns.model.backbones import FLABackboneConfig


MODEL_TYPES = ("gla", "deltanet")


def _model_config_kwargs(model_type: str) -> dict[str, object]:
    if model_type == "gla":
        return {
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_heads": 2,
            "intermediate_size": 32,
            "hidden_act": "swish",
            "norm_eps": 1e-5,
            "use_cache": True,
        }
    if model_type == "deltanet":
        return {
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_heads": 2,
            "intermediate_size": 32,
            "hidden_act": "swish",
            "norm_eps": 1e-5,
            "use_cache": True,
            "use_short_conv": False,
        }
    raise ValueError(f"Unsupported model_type: {model_type}")


def _build_backbone(model_type: str) -> torch.nn.Module:
    config = FLABackboneConfig(
        model_type=model_type,
        config_kwargs=_model_config_kwargs(model_type),
    )
    backbone = config.create_backbone(ninp=8, attention_between_features=False)
    backbone.eval()
    return backbone


def _shallow_copy(obj: object) -> object:
    obj_copy = obj.__class__.__new__(obj.__class__)
    obj_copy.__dict__.update(obj.__dict__)
    return obj_copy


def _repeat_value(value: object, repeat: int) -> object:
    if torch.is_tensor(value):
        return value.repeat_interleave(repeat, dim=0)
    if isinstance(value, tuple):
        return tuple(_repeat_value(item, repeat) for item in value)
    return value


def _repeat_state(state: dict[str, object], repeat: int) -> dict[str, object]:
    return {key: _repeat_value(value, repeat) for key, value in state.items()}


def _repeat_cache(cache_params: object, repeat: int) -> object:
    if torch.is_tensor(cache_params):
        return cache_params.repeat_interleave(repeat, dim=0)
    if hasattr(cache_params, "layers"):
        cache_params_copy = _shallow_copy(cache_params)
        new_layers = []
        for layer in cache_params.layers:
            layer_copy = _shallow_copy(layer)
            state = getattr(layer, "state", None)
            if isinstance(state, dict):
                layer_copy.state = _repeat_state(state, repeat)
            else:
                raise ValueError("Unsupported layer state structure for repetition.")
            new_layers.append(layer_copy)
        cache_params_copy.layers = new_layers
        return cache_params_copy
    if hasattr(cache_params, "states"):
        cache_params_copy = _shallow_copy(cache_params)
        cache_params_copy.states = [
            _repeat_state(state, repeat) if isinstance(state, dict) else state
            for state in cache_params.states
        ]
        return cache_params_copy
    raise ValueError("Unsupported cache_params structure for repetition.")


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_gla_stateless_matches_repeated_cache_outputs_and_grads(model_type: str):
    if not torch.cuda.is_available():
        pytest.skip("FLA backend requires CUDA/Triton for this test.")

    torch.manual_seed(0)
    backbone_stateless = _build_backbone(model_type)
    backbone_reference = _build_backbone(model_type)
    backbone_reference.load_state_dict(backbone_stateless.state_dict())
    device = torch.device("cuda")
    backbone_stateless = backbone_stateless.to(device)
    backbone_reference = backbone_reference.to(device)

    batch_size = 2
    seq_len = 10
    num_tokens = 1
    embed_dim = 8
    train_len = 6
    assert 0 < train_len < seq_len

    x = torch.randn(batch_size, seq_len, num_tokens, embed_dim, device=device)
    x_batched = x.transpose(1, 2).reshape(batch_size * num_tokens, seq_len, embed_dim)
    train_x_base = x_batched[:, :train_len].detach()
    test_x_base = x_batched[:, train_len:].detach()
    test_len = test_x_base.size(1)

    train_x_stateless = train_x_base.clone().requires_grad_(True)
    test_x_stateless = test_x_base.clone().requires_grad_(True)
    _, past_stateless = backbone_stateless._run_fla(train_x_stateless)
    assert past_stateless is not None
    out_stateless = backbone_stateless._run_test_with_cache(test_x_stateless, past_stateless)
    out_stateless.sum().backward()

    train_x_ref = train_x_base.clone().requires_grad_(True)
    test_x_ref = test_x_base.clone().requires_grad_(True)
    _, past_ref = backbone_reference._run_fla(train_x_ref)
    assert past_ref is not None
    repeated_cache = _repeat_cache(past_ref, test_len)
    test_x_flat = test_x_ref.contiguous().view(batch_size * test_len, 1, embed_dim)
    out_ref = backbone_reference.fla(
        inputs_embeds=test_x_flat,
        past_key_values=repeated_cache,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    out_ref = out_ref.view(batch_size, test_len, embed_dim)
    out_ref.sum().backward()

    if model_type == "deltanet":
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-6, 1e-6

    torch.testing.assert_close(out_stateless, out_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(train_x_stateless.grad, train_x_ref.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(test_x_stateless.grad, test_x_ref.grad, rtol=rtol, atol=atol)