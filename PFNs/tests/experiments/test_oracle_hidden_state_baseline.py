import pytest
import torch

pytest.importorskip("fla")

from pfns.experiments.model_benchmarks.oracle_hidden_state_baseline import (
    OracleHiddenStateBaseline,
    OracleHiddenStateConfig,
)
from pfns.model.tabular_model import TabularModel
from tests.model.fla_test_utils import build_fla_backbone, fla_hidden_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA backend requires CUDA/Triton.")
def test_oracle_hidden_state_baseline_reduces_train_loss():
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_features = 4

    backbone = build_fla_backbone("deltanet", size="small", train=False).to(device)
    model = TabularModel(
        transformer_layers=backbone,
        ninp=fla_hidden_size("deltanet", size="small"),
        nhid=32,
        attention_between_features=False,
        features_per_group=num_features,
        batch_first=True,
    ).to(device)
    model.criterion = torch.nn.MSELoss(reduction="none")
    model.eval()

    oracle = OracleHiddenStateBaseline(
        base_model=model,
        optimization_config=OracleHiddenStateConfig(
            num_epochs=20,
            lr=0.1,
            patience=8,
            tolerance=0.0,
            query_batch_size=2,
        ),
    ).to(device)

    train_x = torch.randn(1, 4, num_features, device=device)
    train_y = torch.randn(1, 4, 1, device=device)

    with torch.no_grad():
        initial_state = model.incontext_fit(train_x, train_y)
        initial_pred = model.incontext_predict(
            initial_state,
            test_x=train_x,
            only_return_standard_out=True,
        )
        initial_loss = torch.nn.functional.mse_loss(initial_pred, train_y).item()

    optimized_state = oracle.incontext_fit(train_x, train_y)
    with torch.no_grad():
        optimized_pred = oracle.incontext_predict(
            optimized_state,
            test_x=train_x,
            only_return_standard_out=True,
        )
        optimized_loss = torch.nn.functional.mse_loss(optimized_pred, train_y).item()

    assert oracle.requires_grad_during_eval is True
    assert optimized_state.size_bytes() > 0
    assert optimized_loss <= initial_loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA backend requires CUDA/Triton.")
def test_oracle_hidden_state_seed_controls_split_and_minibatch_order():
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_features = 4

    backbone = build_fla_backbone("deltanet", size="small", train=False).to(device)
    model = TabularModel(
        transformer_layers=backbone,
        ninp=fla_hidden_size("deltanet", size="small"),
        nhid=32,
        attention_between_features=False,
        features_per_group=num_features,
        batch_first=True,
    ).to(device)
    model.criterion = torch.nn.MSELoss(reduction="none")
    model.eval()

    oracle = OracleHiddenStateBaseline(
        base_model=model,
        optimization_config=OracleHiddenStateConfig(
            num_epochs=1,
            lr=0.1,
            patience=2,
            tolerance=0.0,
            query_batch_size=2,
            selection_fraction=0.25,
            selection_seed=123,
        ),
    ).to(device)

    train_x = torch.randn(1, 8, num_features, device=device)
    train_y = torch.randn(1, 8, 1, device=device)

    optimize_x_1, optimize_y_1, selection_x_1, selection_y_1, _ = oracle._train_and_val_split(train_x, train_y)
    optimize_x_2, optimize_y_2, selection_x_2, selection_y_2, _ = oracle._train_and_val_split(train_x, train_y)

    assert torch.equal(optimize_x_1, optimize_x_2)
    assert torch.equal(optimize_y_1, optimize_y_2)
    assert torch.equal(selection_x_1, selection_x_2)
    assert torch.equal(selection_y_1, selection_y_2)

    generator_1 = torch.Generator(device=device)
    generator_1.manual_seed(oracle.optimization_config.selection_seed + 1)
    generator_2 = torch.Generator(device=device)
    generator_2.manual_seed(oracle.optimization_config.selection_seed + 1)
    permutation_1 = torch.randperm(optimize_x_1.shape[1], device=device, generator=generator_1)
    permutation_2 = torch.randperm(optimize_x_1.shape[1], device=device, generator=generator_2)

    batch_x_1, batch_y_1, next_perm_1, next_offset_1 = oracle._sample_query_batch(
        x=optimize_x_1,
        y=optimize_y_1,
        permutation=permutation_1,
        perm_offset=0,
        query_batch_size=2,
        generator=generator_1,
    )
    batch_x_2, batch_y_2, next_perm_2, next_offset_2 = oracle._sample_query_batch(
        x=optimize_x_1,
        y=optimize_y_1,
        permutation=permutation_2,
        perm_offset=0,
        query_batch_size=2,
        generator=generator_2,
    )

    assert torch.equal(batch_x_1, batch_x_2)
    assert torch.equal(batch_y_1, batch_y_2)
    assert torch.equal(next_perm_1, next_perm_2)
    assert next_offset_1 == next_offset_2
