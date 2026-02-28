import jax
import gpjax as gpx
import jax.numpy as jnp
from .algorithms import Gram_XX_jit, Cross_XZ_jit


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

    def gram(self, kernel, X, X_size):
        return self._gram(kernel, X, X_size)

    def cross_covariance(self, kernel, X, Z, X_size, Z_size):
        return self._cross_covariance(kernel, X, Z, X_size, Z_size)

    def diagonal(self, kernel, X, X_size):
        return self._diagonal(kernel, X, X_size)
