import pytest
import torch

from pfns.model.backbones import (
    LinearAttentionBackboneConfig,
    RebasedBackboneConfig,
    TransformerBackboneConfig,
)
from pfns.model.tabular_model import TabularModel
from tests.model.fla_test_utils import (
    FLA_MODEL_TYPES,
    build_fla_backbone,
    fla_hidden_size,
    fla_tolerances,
)


def _build_model(
    *,
    backbone: torch.nn.Module,
    ninp: int,
    num_features: int,
    attention_between_features: bool,
    device: torch.device,
) -> TabularModel:
    features_per_group = 1 if attention_between_features else num_features
    model = TabularModel(
        transformer_layers=backbone,
        ninp=ninp,
        nhid=64,
        attention_between_features=attention_between_features,
        features_per_group=features_per_group,
        batch_first=True,
    )
    model.eval()
    return model.to(device)


def _sample_batch(
    *,
    device: torch.device,
    batch_size: int = 2,
    train_len: int = 7,
    test_len: int = 4,
    num_features: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_x = torch.randn(batch_size, train_len, num_features, device=device)
    train_y = torch.randn(batch_size, train_len, 1, device=device)
    test_x = torch.randn(batch_size, test_len, num_features, device=device)
    return train_x, train_y, test_x


def _assert_incontext_fit_predict_matches_forward(
    model: TabularModel,
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    train_len: int,
    rtol: float,
    atol: float,
) -> None:
    with torch.no_grad():
        baseline = model(
            x=train_x,
            y=train_y,
            test_x=test_x,
            single_eval_pos=train_len,
            only_return_standard_out=True,
        )
        state = model.incontext_fit(
            x=train_x,
            y=train_y,
        )
        split_out = model.incontext_predict(
            state,
            test_x=test_x,
            only_return_standard_out=True,
        )

    torch.testing.assert_close(split_out, baseline, rtol=rtol, atol=atol)
    assert state.size_bytes() >= 0


@pytest.mark.parametrize(
    "case_name, backbone_cfg, attention_between_features, ninp",
    [
        pytest.param(
            "transformer",
            TransformerBackboneConfig(nhead=2, nhid=64, nlayers=2),
            True,
            32,
            id="transformer",
        ),
        pytest.param(
            "transformer_rope",
            TransformerBackboneConfig(
                nhead=2,
                nhid=64,
                nlayers=2,
                layer_kwargs={"item_attention_use_rope": True},
            ),
            True,
            32,
            id="transformer_rope",
        ),
        pytest.param(
            "linear_attention",
            LinearAttentionBackboneConfig(
                nlayers=2,
                nhead=2,
                mlp_hidden_dim=64,
                dropout_prob=0.0,
            ),
            False,
            32,
            id="linear_attention",
        ),
        pytest.param(
            "linear_attention_causal",
            LinearAttentionBackboneConfig(
                nlayers=2,
                nhead=2,
                mlp_hidden_dim=64,
                dropout_prob=0.0,
                layer_kwargs={"causal": True},
            ),
            False,
            32,
            id="linear_attention_causal",
        ),
        pytest.param(
            "rebased",
            RebasedBackboneConfig(
                nlayers=2,
                mlp_hidden_dim=64,
                num_heads=2,
                dropout=0.0,
            ),
            False,
            32,
            id="rebased",
        ),
    ],
)
def test_incontext_fit_predict_matches_forward_non_fla(
    case_name: str,
    backbone_cfg: TransformerBackboneConfig
    | LinearAttentionBackboneConfig
    | RebasedBackboneConfig,
    attention_between_features: bool,
    ninp: int,
) -> None:
    torch.manual_seed(0)
    if case_name == "rebased" and not torch.cuda.is_available():
        pytest.skip("Rebased feature map path requires CUDA/Triton.")

    device = torch.device("cuda" if case_name == "rebased" else "cpu")
    num_features = 4
    train_len = 7

    backbone = backbone_cfg.create_backbone(
        ninp=ninp,
        attention_between_features=attention_between_features,
    )
    model = _build_model(
        backbone=backbone,
        ninp=ninp,
        num_features=num_features,
        attention_between_features=attention_between_features,
        device=device,
    )

    train_x, train_y, test_x = _sample_batch(
        device=device,
        train_len=train_len,
        num_features=num_features,
    )

    _assert_incontext_fit_predict_matches_forward(
        model,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        train_len=train_len,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("model_type", FLA_MODEL_TYPES)
def test_incontext_fit_predict_matches_forward_fla(model_type: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("FLA equivalence test requires CUDA.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_features = 4
    train_len = 7
    ninp = fla_hidden_size(model_type, size="equivalence")
    backbone = build_fla_backbone(
        model_type,
        size="equivalence",
        sequence_mode="Comb_ST",
        train=False,
    )
    model = _build_model(
        backbone=backbone,
        ninp=ninp,
        num_features=num_features,
        attention_between_features=False,
        device=device,
    )

    train_x, train_y, test_x = _sample_batch(
        device=device,
        train_len=train_len,
        num_features=num_features,
    )

    rtol, atol = fla_tolerances(model_type, default=(1e-5, 1e-5))
    _assert_incontext_fit_predict_matches_forward(
        model,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        train_len=train_len,
        rtol=rtol,
        atol=atol,
    )
