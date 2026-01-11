import jax
import jax.numpy as jnp
import gpjax as gpx
from gpjax.linalg import Dense, psd
from gpjax.parameters import Real, Parameter, NonNegativeReal
from .algorithms import Gram_XX_jit, Cross_XZ_jit, Diag_XX_jit


class CustomComputeEngine(gpx.kernels.computations.AbstractKernelComputation):
    """Custom compute engine class for the signature kernel.
    """
    def _gram(self, kernel, X, X_size):
        lengthscales = kernel.lengthscales
        amplitude = kernel.amplitude
        weights = kernel.weights
        return Gram_XX_jit(
            X, X_size, kernel.n_timesteps, kernel.n_nontrivial_levels, lengthscales.get_value(), amplitude.get_value(), weights.get_value()
            )

    def _cross_covariance(self, kernel, X, Z, X_size, Z_size):
        lengthscales = kernel.lengthscales
        amplitude = kernel.amplitude
        weights = kernel.weights
        return Cross_XZ_jit(
            X, Z, X_size, Z_size, kernel.n_timesteps, kernel.n_nontrivial_levels, lengthscales.get_value(), amplitude.get_value(), weights.get_value()
            )

    def _diagonal(self, kernel, X, X_size):
        lengthscales = kernel.lengthscales
        amplitude = kernel.amplitude
        weights = kernel.weights
        return Diag_XX_jit(
            X, X_size, kernel.n_timesteps, kernel.n_nontrivial_levels, lengthscales.get_value(), amplitude.get_value(), weights.get_value()
            )

    def gram(self, kernel, X, X_size):
        return self._gram(kernel, X, X_size)

    def cross_covariance(self, kernel, X, Z, X_size, Z_size):
        return self._cross_covariance(kernel, X, Z, X_size, Z_size)

    def diagonal(self, kernel, X, X_size):
        return self._diagonal(kernel, X, X_size)



class SignatureKernel(gpx.kernels.AbstractKernel):
    """Custom kernel class for the signature kernel.

    Note that we do not define pointwise kernel evaluations.
    """
    def __init__(self, n_dimensions, n_timesteps, n_nontrivial_levels, lengthscales=None, amplitude=None, weights=None, compute_engine=CustomComputeEngine):
        self.n_dimensions = n_dimensions
        self.n_timesteps = n_timesteps
        self.n_nontrivial_levels = n_nontrivial_levels
        self.compute_engine = compute_engine() if isinstance(compute_engine, type) else compute_engine
        self.lengthscales = NonNegativeReal(lengthscales if lengthscales is not None else self.default_lengthscales())
        self.amplitude = NonNegativeReal(amplitude if amplitude is not None else self.default_amplitude())
        self.weights = NonNegativeReal(weights if weights is not None else self.default_weights())
        super().__init__(active_dims=slice(None), n_dims=n_dimensions * n_timesteps, compute_engine=self.compute_engine)

    def default_lengthscales(self):
        C = jnp.sqrt(2*self.n_dimensions)
        return C * jnp.ones(self.n_dimensions)

    def default_amplitude(self):
        return jnp.array([1.0])

    def default_weights(self):
        return jnp.ones(self.n_nontrivial_levels+1)

    def gram(self, X):
        X_size = X.shape[0]
        return psd(Dense(self.compute_engine.gram(self, X, X_size)))

    def cross_covariance(self, X, Z):
        X_size = X.shape[0]
        Z_size = Z.shape[0]
        return self.compute_engine.cross_covariance(self, X, Z, X_size, Z_size)

    def diagonal(self, X):
        X_size = X.shape[0]
        return self.compute_engine.diagonal(self, X, X_size)

    def __call__(self, X, Z):
        raise NotImplementedError("Pointwise kernel evaluation is not implemented.")
