import h5py
import jax
import jax.numpy as jnp
import flax.serialization as flax_ser

from functools import partial
from flax import linen as nn

from fastar.path_utils import get_data_path

# =============================================================================
# PCA-based Neural Network Model Definition
# =============================================================================
class PCARegressor(nn.Module):
    """
    Simple feed-forward neural network for predicting PCA coefficients
    from stellar parameters.

    Attributes
    ----------
    output_dim : int
        Number of PCA components to output.
    activation_type : str
        Type of activation function ('relu', 'tanh', 'gelu').
    """
    output_dim: int = 16
    activation_type: str = 'gelu'

    @nn.compact
    def __call__(self, x):
        act = {'relu': nn.relu, 'tanh': nn.tanh, 'gelu': nn.gelu}[self.activation_type]
        x = nn.Dense(64)(x)
        x = act(x)
        x = nn.Dense(128)(x)
        x = act(x)
        x = nn.Dense(128)(x)
        x = act(x)
        x = nn.Dense(64)(x)
        x = act(x)
        x = nn.Dense(self.output_dim)(x)
        return x

# =============================================================================
# Semi-Resolved Population Synthesizer Class
# =============================================================================
class StellarSynthesizer:
    """
    Class for generating stellar SED predictions with a PCA-based model
    """
    def __init__(self, model_label=None, imf_function=None):
        self.npc = 16
        self.activation_type = 'gelu'

        if model_label is None:
            self.rlabel = f'_spec'

        if model_label == 'phot':
            self.rlabel = f'_phot'

        self._load_model()

    def _load_model(self):
        """
        Load trained PCA regressor, scalers, and PCA components.
        """
        model = PCARegressor(output_dim=self.npc, activation_type=self.activation_type)
        with open(get_data_path(f"pca_regressor{self.rlabel}.flax",subdir="aux"), "rb") as f:
            self.params = flax_ser.from_bytes(model.init(jax.random.PRNGKey(0), jnp.ones((1, 3))), f.read())
        self.model = model

        with h5py.File(get_data_path(f"training_artifacts{self.rlabel}.h5",subdir="aux"), "r") as f:
            self.scaler_X_mean = f['scaler_X/mean_'][:]
            self.scaler_X_scale = f['scaler_X/scale_'][:]
            self.scaler_Y_mean = f['scaler_Y/mean_'][:]
            self.scaler_Y_scale = f['scaler_Y/scale_'][:]
            self.pca_components = f['pca/components_'][:]
            self.pca_mean = f['pca/mean_'][:]
            self.mean_spectrum = f['mean_spectrum'][:]
            self.wave = f['wave'][:]
    
    @partial(jax.jit, static_argnames=['self'])
    def _softplus(self, x, beta=100.0):
        """
        Smooth activation function with soft floor to prevent
        any negative flux in the spectrum of extreme stars
        """
        return (1.0/beta) * jnp.logaddexp(0.0, beta*x)
    
    @partial(jax.jit, static_argnames=['self'])
    def stellar_spectrum(self, logg, teff, fmet):
        """
        Predict stellar spectra given logg, Teff, and [Fe/H] using the 
        PCA regressor.
        """
        inputs = jnp.stack([logg, teff, fmet], axis=-1)
        input_scaled = (inputs - self.scaler_X_mean) / self.scaler_X_scale
        pca_scaled = self.model.apply(self.params, input_scaled)
        pca_coeffs = pca_scaled * self.scaler_Y_scale + self.scaler_Y_mean
        spectra = jnp.dot(pca_coeffs, self.pca_components) + self.pca_mean + self.mean_spectrum

        return self._softplus(spectra)