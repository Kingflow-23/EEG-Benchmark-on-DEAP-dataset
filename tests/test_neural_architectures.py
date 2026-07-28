"""Topology regression tests for production-derived neural architectures."""

import pytest

from src.models.torch_tabular import _make_network


torch = pytest.importorskip("torch")


def _parameters(model) -> int:
    """Count trainable parameters in a test network."""
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


def test_neural_architecture_parameter_counts():
    """Guard the exact MLP, CNN, and published LSTM layer dimensions."""
    assert _parameters(_make_network("feature_mlp", 160)) == 83_394
    assert _parameters(_make_network("band_electrode_cnn", 160)) == 7_042
    assert _parameters(_make_network("fft_lstm", 160)) == 789_298


@pytest.mark.parametrize("name", ["feature_mlp", "band_electrode_cnn",
                                  "fft_lstm", "ft_transformer"])
def test_neural_models_emit_two_logits(name):
    """Every neural candidate must satisfy the common binary-logit contract."""
    model = _make_network(name, 160).eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 160))
    assert output.shape == (2, 2)
