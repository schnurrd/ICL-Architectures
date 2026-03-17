import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from pfns.model import encoders, tabular_model
from pfns.model.tabular_model import isolate_torch_rng, TabularModel
from pfns.model.backbones import TransformerBackboneConfig
from torch.nn import CrossEntropyLoss


class SimpleStyleEncoder(nn.Module):
    def __init__(self, ninp):
        super().__init__()
        self.linear = nn.Linear(5, ninp)

    def forward(self, x):
        if x.ndim == 2:
            return self.linear(x)
        elif x.ndim == 3:  # shape: batch size x num_features x style_dim
            return self.linear(x).sum(dim=1)


@pytest.fixture(params=[True, False], ids=["batch_first_True", "batch_first_False"])
def batch_first_setting(request):
    """Provides True for batch_first=True and False for batch_first=False."""
    return request.param


@pytest.fixture
def sample_data(batch_first_setting):
    is_batch_first = batch_first_setting
    batch_size = 4
    seq_len_train = 10
    seq_len_test = 5
    num_features = 3

    if is_batch_first:
        train_x = torch.randn(batch_size, seq_len_train, num_features)
        train_y = torch.randn(batch_size, seq_len_train, 1)
        test_x = torch.randn(batch_size, seq_len_test, num_features)
    else:
        train_x = torch.randn(seq_len_train, batch_size, num_features)
        train_y = torch.randn(seq_len_train, batch_size, 1)
        test_x = torch.randn(seq_len_test, batch_size, num_features)

    return {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "batch_size": batch_size,
        "seq_len_train": seq_len_train,
        "seq_len_test": seq_len_test,
        "num_features": num_features,
        "batch_first": is_batch_first,  # Store the flag
    }


def test_transformer_init():
    """Test basic initialization of the transformer."""
    backbone = TransformerBackboneConfig(nhead=4, nhid=256, nlayers=6).create_backbone(ninp=64, attention_between_features=True)
    transformer = TabularModel(transformer_layers=backbone, ninp=64, nhid=256)

    assert transformer.ninp == 64
    assert transformer.nhid == 256
    assert len(transformer.transformer_layers.layer_stack.layers) == 6


def test_transformer_forward(sample_data):
    """Test basic forward pass with default parameters."""
    model_batch_first = sample_data["batch_first"]
    backbone = TransformerBackboneConfig(nhead=2, nhid=64, nlayers=2).create_backbone(ninp=32, attention_between_features=True)
    transformer_model = TabularModel(
        transformer_layers=backbone,
        ninp=32, nhid=64, batch_first=model_batch_first
    )

    output = transformer_model(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
    )

    assert isinstance(output, torch.Tensor)
    if model_batch_first:
        assert output.shape == (
            sample_data["batch_size"],
            sample_data["seq_len_test"],
            1,
        )
    else:
        assert output.shape == (
            sample_data["seq_len_test"],
            sample_data["batch_size"],
            1,
        )


def test_add_embeddings_normal_rand_vec():
    """Test that add_embeddings adds the same embeddings when using normal_rand_vec."""
    # Create a transformer with normal_rand_vec feature positional embedding
    backbone = TransformerBackboneConfig(nhead=4, nhid=256, nlayers=6).create_backbone(ninp=64, attention_between_features=True)
    transformer = TabularModel(
        transformer_layers=backbone,
        ninp=64,
        nhid=256,
        feature_positional_embedding="normal_rand_vec",
        seed=42,
    )

    # Create two identical zero tensors
    batch_size = 2
    seq_len = 3
    num_groups = 4
    emsize = 64

    x1 = torch.zeros((batch_size, seq_len, num_groups, emsize), device="cpu")
    y1 = torch.zeros((batch_size, seq_len, emsize), device="cpu")

    x2 = torch.zeros((batch_size, seq_len, num_groups, emsize), device="cpu")
    y2 = torch.zeros((batch_size, seq_len, emsize), device="cpu")

    # Add embeddings to both sets of tensors
    x1_out, y1_out = transformer.add_embeddings(
        x1,
        y1,
        num_features=num_groups,
        seq_len=seq_len,
        cache_embeddings=False,
        use_cached_embeddings=False,
    )

    x2_out, y2_out = transformer.add_embeddings(
        x2,
        y2,
        num_features=num_groups,
        seq_len=seq_len,
        cache_embeddings=False,
        use_cached_embeddings=False,
    )

    # Check that the embeddings are the same
    assert torch.allclose(
        x1_out, x2_out
    ), "Embeddings added to x should be identical with the same seed"
    assert torch.allclose(
        y1_out, y2_out
    ), "Embeddings added to y should be identical with the same seed"

    # Check that the embeddings are not zero (they were actually added)
    assert not torch.allclose(
        x1_out, torch.zeros_like(x1_out)
    ), "Embeddings should change the zero tensor"

    x3 = torch.zeros((batch_size, seq_len, num_groups, emsize), device="cpu")
    y3 = torch.zeros((batch_size, seq_len, emsize), device="cpu")

    transformer.seed = 43
    x3_out, y3_out = transformer.add_embeddings(
        x3,
        y3,
        num_features=num_groups,
        seq_len=seq_len,
        cache_embeddings=False,
        use_cached_embeddings=False,
    )

    # Check that embeddings are different with different seed
    assert not torch.allclose(
        x1_out, x3_out
    ), "Embeddings should be different with different seeds"


def test_isolate_torch_rng():
    """Test that isolate_torch_rng properly isolates the random number generator state."""

    # Generate a random tensor before using isolate_torch_rng
    torch.manual_seed(123)
    tensor_before1 = torch.rand(5)
    tensor_before2 = torch.rand(5)
    torch.manual_seed(123)
    tensor_before_1 = torch.rand(5)

    # Sanity check that seeding works
    assert torch.allclose(tensor_before1, tensor_before_1)

    # Use isolate_torch_rng with a different seed
    with isolate_torch_rng(seed=123, device=torch.device("cpu")):
        # Check if seed correctly set
        isolated_tensor1 = torch.rand(5)
        isolated_tensor2 = torch.rand(5)

        # These should be identical
        assert torch.allclose(isolated_tensor1, tensor_before1)
        assert torch.allclose(isolated_tensor1, tensor_before_1)
        assert torch.allclose(isolated_tensor2, tensor_before2)

    # RNG state restorement
    tensor_after_2 = torch.rand(5)

    assert torch.allclose(tensor_before2, tensor_after_2)
    assert torch.allclose(tensor_after_2, isolated_tensor2)


def test_feature_positional_embeddings(sample_data):
    """Test different feature positional embedding options."""
    embedding_types = [
        "normal_rand_vec",
        "uni_rand_vec",
        "learned",
        "subspace",
        None,
    ]
    model_batch_first = sample_data["batch_first"]

    for emb_type in embedding_types:
        backbone = TransformerBackboneConfig(nhead=2, nhid=64, nlayers=2).create_backbone(ninp=32, attention_between_features=True)
        transformer_model = TabularModel(
            transformer_layers=backbone,
            ninp=32,
            nhid=64,
            feature_positional_embedding=emb_type,
            seed=42,
            batch_first=model_batch_first,
        )

        output = transformer_model(
            x=sample_data["train_x"],
            y=sample_data["train_y"],
            test_x=sample_data["test_x"],
        )

        assert isinstance(output, torch.Tensor)
        if model_batch_first:
            assert output.shape == (
                sample_data["batch_size"],
                sample_data["seq_len_test"],
                1,
            )
        else:
            assert output.shape == (
                sample_data["seq_len_test"],
                sample_data["batch_size"],
                1,
            )


def test_features_per_group(sample_data):
    """Test the features_per_group parameter."""
    model_batch_first = sample_data["batch_first"]
    # Set features_per_group=3 to match the number of features in sample data
    backbone = TransformerBackboneConfig(nhead=2, nhid=64, nlayers=2).create_backbone(ninp=32, attention_between_features=True)
    transformer_model = TabularModel(
        transformer_layers=backbone,
        ninp=32,
        nhid=64,
        features_per_group=3,
        batch_first=model_batch_first,
    )

    output = transformer_model(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
    )

    assert isinstance(output, torch.Tensor)
    if model_batch_first:
        assert output.shape == (
            sample_data["batch_size"],
            sample_data["seq_len_test"],
            1,
        )
    else:
        assert output.shape == (
            sample_data["seq_len_test"],
            sample_data["batch_size"],
            1,
        )


@torch.inference_mode()
def test_cache_trainset_representation(sample_data):
    """Test caching of trainset representations."""
    model_batch_first = sample_data["batch_first"]
    backbone = TransformerBackboneConfig(nhead=2, nhid=64, nlayers=2).create_backbone(ninp=32, attention_between_features=True)
    transformer_model = TabularModel(
        transformer_layers=backbone,
        ninp=32,
        nhid=64,
        cache_trainset_representation=True,
        batch_first=model_batch_first,
    )

    # First forward pass should cache the representations
    output1 = transformer_model(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
    )

    # Second forward pass should use the cached representations
    # For cached representation, x and y are None, test_x provides the query points.
    # The format of test_x should match the model's batch_first setting.
    output2 = transformer_model(x=None, y=None, test_x=sample_data["test_x"])

    assert torch.allclose(output1, output2, atol=1e-7)

    # Clear cache and results should be different
    transformer_model.empty_trainset_representation_cache()

    # After clearing, we need to provide train data again
    output3 = transformer_model(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
    )

    assert torch.allclose(output1, output3, atol=1e-7)  # Should be deterministic


def test_decoder_dict(sample_data):
    """Test custom decoder dictionary."""

    class CustomDecoder(nn.Module):
        def __init__(self, ninp, nhid, nout):
            super().__init__()
            self.linear1 = nn.Linear(ninp, nhid)
            self.linear2 = nn.Linear(nhid, nout)

        def forward(self, x):
            return self.linear2(torch.relu(self.linear1(x)))

    decoder_dict = {
        "standard": (None, 1),  # Default MLP
        "custom": (CustomDecoder, 2),  # Custom decoder with 2 outputs
    }

    model_batch_first = sample_data["batch_first"]

    backbone = TransformerBackboneConfig(nhead=2, nhid=64, nlayers=2).create_backbone(ninp=32, attention_between_features=True)
    transformer_model = TabularModel(
        transformer_layers=backbone,
        ninp=32,
        nhid=64,
        decoder_dict=decoder_dict,
        batch_first=model_batch_first,
    )

    # Get all outputs
    outputs = transformer_model(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
        only_return_standard_out=False,
    )

    assert isinstance(outputs, dict)
    assert "standard" in outputs
    assert "custom" in outputs

    if model_batch_first:
        expected_shape_standard = (
            sample_data["batch_size"],
            sample_data["seq_len_test"],
            1,
        )
        expected_shape_custom = (
            sample_data["batch_size"],
            sample_data["seq_len_test"],
            2,
        )
    else:
        expected_shape_standard = (
            sample_data["seq_len_test"],
            sample_data["batch_size"],
            1,
        )
        expected_shape_custom = (
            sample_data["seq_len_test"],
            sample_data["batch_size"],
            2,
        )
    assert outputs["standard"].shape == expected_shape_standard
    assert outputs["custom"].shape == expected_shape_custom


def test_style_encoder(sample_data):
    """Test the style encoder functionality."""
    style_encoder_module = SimpleStyleEncoder(32)
    model_batch_first = sample_data["batch_first"]

    backbone = TransformerBackboneConfig(nhead=2, nhid=64, nlayers=2).create_backbone(ninp=32, attention_between_features=True)
    transformer_model = TabularModel(
        transformer_layers=backbone,
        ninp=32,
        nhid=64,
        style_encoder=style_encoder_module,
        batch_first=model_batch_first,
    )

    # Create style vectors for each batch (always batch-first for style)
    style = torch.randn(sample_data["batch_size"], 5)

    output = transformer_model(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
        style=style,
    )

    assert isinstance(output, torch.Tensor)
    if model_batch_first:
        expected_shape = (
            sample_data["batch_size"],
            sample_data["seq_len_test"],
            1,
        )
    else:
        expected_shape = (
            sample_data["seq_len_test"],
            sample_data["batch_size"],
            1,
        )
    assert output.shape == expected_shape

    # Test per-feature style vectors (always batch-first for style)
    feature_style = torch.randn(
        sample_data["batch_size"], sample_data["num_features"], 5
    )

    output2 = transformer_model(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
        style=feature_style,
    )

    assert isinstance(output2, torch.Tensor)
    assert output2.shape == expected_shape


def test_y_style_encoder(sample_data):
    """Test the y_style encoder functionality."""
    style_encoder_module = SimpleStyleEncoder(32)
    model_batch_first = sample_data["batch_first"]

    backbone = TransformerBackboneConfig(nhead=2, nhid=64, nlayers=2).create_backbone(ninp=32, attention_between_features=True)
    transformer_model = TabularModel(
        transformer_layers=backbone,
        ninp=32,
        nhid=64,
        style_encoder=style_encoder_module,  # y_style_encoder requires attention_between_features=True
        y_style_encoder=style_encoder_module,
        attention_between_features=True,  # Required for y_style_encoder
        batch_first=model_batch_first,
    )

    # Create y_style vectors for each batch (always batch-first for style)
    y_style = torch.randn(sample_data["batch_size"], 5)

    output = transformer_model(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
        y_style=y_style,
    )

    assert isinstance(output, torch.Tensor)
    if model_batch_first:
        expected_shape = (
            sample_data["batch_size"],
            sample_data["seq_len_test"],
            1,
        )
    else:
        expected_shape = (
            sample_data["seq_len_test"],
            sample_data["batch_size"],
            1,
        )
    assert output.shape == expected_shape

    # Test with both style and y_style
    style = torch.randn(sample_data["batch_size"], 5)

    output2 = transformer_model(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
        style=style,
        y_style=y_style,
    )

    assert isinstance(output2, torch.Tensor)
    assert output2.shape == expected_shape

    # Test per-feature style with both style and y_style
    feature_style = torch.randn(
        sample_data["batch_size"], sample_data["num_features"], 5
    )

    output3 = transformer_model(
        x=sample_data["train_x"],
        y=sample_data["train_y"],
        test_x=sample_data["test_x"],
        style=feature_style,
        y_style=y_style,
    )

    assert isinstance(output3, torch.Tensor)
    assert output3.shape == expected_shape


@pytest.mark.parametrize(
    "multiquery_item_attention_for_test_set",
    [False, True],
)
@pytest.mark.parametrize(
    "model_batch_first_setting",
    [True, False],
    ids=["model_BF_True", "model_BF_False"],
)
@torch.inference_mode()
def test_separate_train_inference(
    multiquery_item_attention_for_test_set, model_batch_first_setting
):
    backbone = TransformerBackboneConfig().create_backbone(ninp=tabular_model.DEFAULT_EMSIZE, attention_between_features=True)
    model = tabular_model.TabularModel(
        transformer_layers=backbone,
        encoder=encoders.SequentialEncoder(
            encoders.InputNormalizationEncoderStep(
                normalize_on_train_only=True,
                normalize_x=True,
                remove_outliers=True,
            ),
            encoders.LinearInputEncoderStep(
                num_features=1,  # This encoder expects 1 feature after grouping
                emsize=tabular_model.DEFAULT_EMSIZE,
                in_keys=["main"],
                out_keys=["output"],
            ),
        ),
        batch_first=model_batch_first_setting,
    )

    for p in model.parameters():
        p += 0.01  # make it more interesting, not anymore mean 0

    model.feature_positional_embedding = None  # 'subspace'
    for layer in model.transformer_layers.layers:
        layer.multiquery_item_attention_for_test_set = (
            multiquery_item_attention_for_test_set
        )

    model.cache_trainset_representation = True
    model.reset_save_peak_mem_factor(None)
    model.empty_trainset_representation_cache()

    device = "cpu"

    n_train = 10
    n_features = 10  # Should match features_per_group for this encoder setup
    n_test = 3
    batch_size = 2
    # Create data as sequence-first initially
    x_train_sf = torch.normal(
        0.0,
        2.0,
        size=(n_train, batch_size, n_features),
        device=device,
    )
    y_sf = (x_train_sf[:, :, :1] > 1.0).float().to(device).to(torch.float)
    x_test_sf = torch.normal(
        0.0,
        1.0,
        size=(n_test, batch_size, n_features),
        device=device,
    )

    # Prepare data based on model's batch_first setting
    if model_batch_first_setting:
        x_train_model = x_train_sf.transpose(0, 1)
        y_model = y_sf.transpose(0, 1)
        x_test_model = x_test_sf.transpose(0, 1)
    else:
        x_train_model = x_train_sf
        y_model = y_sf
        x_test_model = x_test_sf

    torch.manual_seed(12345)
    # Pass only training part (x_train_model, y_model without test part)
    # The model's forward method handles combining x and test_x if test_x is provided.
    # Here, we are testing the two-step inference:
    # 1. Prime with training data (x_train_model, y_model up to n_train)
    # 2. Infer with test data (x_test_model, y=None)
    model(
        x_train_model,
        y_model[:, :n_train] if model_batch_first_setting else y_model[:n_train, :],
    )
    logits1 = model(x_test_model, None)  # y is None for inference on x_test_model

    torch.manual_seed(12345)
    # Single call with train and test data
    # model's forward will split y internally for _forward's single_eval_pos
    logits1a = model(x_train_model, y_model, x_test_model)

    assert logits1.float() == pytest.approx(
        logits1a.float(), abs=1e-5
    ), f"{logits1} != {logits1a}"


def test_input_normalization_encoder_step_skips_categorical_columns():
    step = encoders.InputNormalizationEncoderStep(
        normalize_on_train_only=True,
        normalize_x=True,
        remove_outliers=True,
    )

    x = torch.tensor(
        [
            [[0.0, 1.0], [10.0, 0.0]],
            [[1.0, 3.0], [20.0, 1.0]],
            [[2.0, 5.0], [30.0, 0.0]],
            [[3.0, 7.0], [40.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    categorical_inds = [[1], [1]]

    step._fit(x, single_eval_pos=3, categorical_inds=categorical_inds)
    transformed, = step._transform(x, single_eval_pos=3, categorical_inds=categorical_inds)

    assert torch.equal(transformed[:, :, 1], x[:, :, 1])
    assert not torch.equal(transformed[:, :, 0], x[:, :, 0])


class _RecordingInputNormalizationEncoderStep(encoders.InputNormalizationEncoderStep):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_input = None
        self.last_output = None

    def _transform(self, x: torch.Tensor, single_eval_pos: int, **kwargs):
        self.last_input = x.detach().clone()
        out = super()._transform(x, single_eval_pos, **kwargs)
        self.last_output = out[0].detach().clone()
        return out


@torch.inference_mode()
def test_tabular_model_passes_categorical_inds_to_input_normalization():
    recording_step = _RecordingInputNormalizationEncoderStep(
        normalize_on_train_only=True,
        normalize_x=True,
        remove_outliers=True,
    )
    backbone = TransformerBackboneConfig(
        nhead=2,
        nhid=32,
        nlayers=2,
    ).create_backbone(ninp=tabular_model.DEFAULT_EMSIZE, attention_between_features=True)
    model = tabular_model.TabularModel(
        transformer_layers=backbone,
        encoder=encoders.SequentialEncoder(
            encoders.OrdinalEncoderStep(),
            recording_step,
            encoders.LinearInputEncoderStep(
                num_features=2,
                emsize=tabular_model.DEFAULT_EMSIZE,
                in_keys=["main"],
                out_keys=["output"],
            ),
        ),
        batch_first=True,
        features_per_group=2,
    )

    x_train = torch.tensor(
        [
            [[0.0, 1.0, 10.0, 0.0], [1.0, 3.0, 11.0, 1.0], [2.0, 5.0, 12.0, 0.0]],
            [[3.0, 7.0, 13.0, 1.0], [4.0, 9.0, 14.0, 0.0], [5.0, 11.0, 15.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    y_train = torch.zeros((2, 3, 1), dtype=torch.float32)
    x_test = torch.tensor(
        [
            [[6.0, 13.0, 16.0, 0.0]],
            [[7.0, 15.0, 17.0, 1.0]],
        ],
        dtype=torch.float32,
    )

    _ = model(
        x=x_train,
        y=y_train,
        test_x=x_test,
        categorical_inds=[1, 3],
    )

    assert recording_step.last_input is not None
    assert recording_step.last_output is not None
    assert torch.equal(recording_step.last_input[:, :, 1], recording_step.last_output[:, :, 1])


@pytest.mark.parametrize(
    "attention_between_features",
    [False, True],
)
def test_transformer_overfit(attention_between_features):
    """Test that a tiny transformer can overfit a simple classification task."""
    with isolate_torch_rng(seed=0, device=torch.device("cpu")):
        # Create a tiny transformer for a simple classification task
        batch_size = 3
        seq_len_train = 3  # 3 examples in context
        seq_len_test = 3  # 3 examples to predict
        emsize = 16  # Small embedding size
        num_classes = 3  # 3-way classification

        # Create a tiny transformer
        backbone = TransformerBackboneConfig(nhead=2, nhid=32, nlayers=2).create_backbone(ninp=emsize, attention_between_features=attention_between_features)
        transformer = TabularModel(
            transformer_layers=backbone,
            ninp=emsize,
            nhid=32,
            features_per_group=1,
            seed=42,
            decoder_dict={"standard": (None, num_classes)},
            attention_between_features=attention_between_features,
        )

        # # Add a classification head
        # transformer.decoder_dict = torch.nn.ModuleDict({
        #     "classification": torch.nn.Linear(emsize, num_classes)
        # })

        # Create training data: map input 0,1,2 to class 0,1,2
        x_train = torch.zeros((batch_size, seq_len_train, 1), device="cpu")
        y_train = torch.zeros((batch_size, seq_len_train, 1), device="cpu")

        # Create test data with the same pattern
        x_test = torch.zeros((batch_size, seq_len_test, 1), device="cpu")
        y_test = torch.zeros((batch_size, seq_len_test), device="cpu", dtype=torch.long)

        # Set the first dimension of each feature to 0, 1, or 2 to represent our input
        for i in range(batch_size):
            for j in range(seq_len_train):
                x_train[i, j] = j % num_classes  # Input is 0, 1, 2
                y_train[i, j] = (j + i) % num_classes  # Input is 0, 1, 2

            for j in range(seq_len_test):
                x_test[i, j] = j % num_classes  # Input is 0, 1, 2
                y_test[i, j] = (j + i) % num_classes  # Input is 0, 1, 2

        # Set up optimizer
        optimizer = optim.Adam(transformer.parameters(), lr=0.01)
        criterion = CrossEntropyLoss()

        # Train the model to overfit
        transformer.train()
        for step in range(100):
            optimizer.zero_grad()

            scramble = torch.randperm(seq_len_train)

            # Forward pass
            logits = transformer(
                x=x_train[:, scramble],
                y=y_train[:, scramble],
                test_x=x_test,
            )

            # Calculate loss
            loss = criterion(logits, y_test)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Check if we've overfit
            if step % 20 == 0 or step == 99:
                with torch.no_grad():
                    # Get predictions
                    _, predicted = torch.max(logits, 1)
                    accuracy = (predicted == y_test).float().mean().item()
                    print(f"Step {step}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}")

                    if accuracy == 1.0 and loss.item() < 0.1:
                        print("Successfully overfit the data!")
                        return
    raise Exception("Failed to overfit the data.")
