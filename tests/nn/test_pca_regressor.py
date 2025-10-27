"""
Tests for PCA regressor neural network model.
"""
import pytest
import jax
import jax.numpy as jnp
from fastar.nn.pca_regressor import PCARegressor


@pytest.mark.unit
class TestPCARegressor:
    """Test suite for PCARegressor."""

    def test_pca_regressor_initialization(self):
        """Test that PCARegressor initializes with default parameters."""
        model = PCARegressor()
        assert model is not None, "Model should initialize"
        assert model.output_dim == 16, "Default output_dim should be 16"
        assert model.activation_type == 'gelu', "Default activation should be 'gelu'"

    def test_pca_regressor_custom_parameters(self):
        """Test that PCARegressor accepts custom parameters."""
        model = PCARegressor(output_dim=32, activation_type='relu')
        assert model.output_dim == 32, "Output dim should be 32"
        assert model.activation_type == 'relu', "Activation should be 'relu'"

    def test_pca_regressor_forward_pass(self):
        """Test that PCARegressor forward pass works."""
        model = PCARegressor(output_dim=16, activation_type='gelu')

        # Initialize model
        key = jax.random.PRNGKey(0)
        x = jnp.ones((1, 3))  # Batch of 1, 3 input features
        params = model.init(key, x)

        # Forward pass
        output = model.apply(params, x)

        assert output.shape == (1, 16), f"Output shape should be (1, 16), got {output.shape}"

    def test_pca_regressor_output_finite(self):
        """Test that PCARegressor output is finite."""
        model = PCARegressor(output_dim=16, activation_type='gelu')

        key = jax.random.PRNGKey(0)
        x = jnp.array([[4.0, 5777.0, 0.0]])  # logg, teff, fmet
        params = model.init(key, x)

        output = model.apply(params, x)

        assert jnp.all(jnp.isfinite(output)), "Output should be finite"

    def test_pca_regressor_batch_processing(self):
        """Test that PCARegressor handles batch inputs."""
        model = PCARegressor(output_dim=16, activation_type='gelu')

        key = jax.random.PRNGKey(0)
        batch_size = 10
        x = jnp.ones((batch_size, 3))
        params = model.init(key, x)

        output = model.apply(params, x)

        assert output.shape == (batch_size, 16), f"Output shape should be ({batch_size}, 16)"

    def test_pca_regressor_different_activations(self):
        """Test that different activation functions work."""
        activations = ['relu', 'tanh', 'gelu']

        for act in activations:
            model = PCARegressor(output_dim=16, activation_type=act)
            key = jax.random.PRNGKey(0)
            x = jnp.ones((1, 3))
            params = model.init(key, x)
            output = model.apply(params, x)

            assert output.shape == (1, 16), f"Activation {act} should work"
            assert jnp.all(jnp.isfinite(output)), f"Output with {act} should be finite"
