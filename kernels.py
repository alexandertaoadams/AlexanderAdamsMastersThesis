import jax
import gpjax as gpx
from gpjax.linalg import Dense, psd
from gpjax.parameters import Real, Parameter, NonNegativeReal




class Custom_Compute_Engine(gpx.kernels.computations.AbstractKernelComputation):
    """
    """
    def _gram(self, kernel, X, X_size):
        n_timesteps = kernel.n_timesteps
        n_nontrivial_levels = kernel.n_nontrivial_levels
        weights = kernel.weights
        separate_levels = Gram_XX_jit(X, X_size, n_timesteps, n_nontrivial_levels)
        return jnp.tensordot(weights, separate_levels, axes=([0], [0]))

    def _cross_covariance(self, kernel, X, Z, X_size, Z_size):
        n_timesteps = kernel.n_timesteps
        n_nontrivial_levels = kernel.n_nontrivial_levels
        return Cross_XZ_jit(X, Z, X_size, Z_size, n_timesteps, n_nontrivial_levels)

    def _diagonal(self, kernel, X, X_size):
        n_timesteps = kernel.n_timesteps
        n_nontrivial_levels = kernel.n_nontrivial_levels
        return Diag_XX_jit(X, X_size, n_timesteps, n_nontrivial_levels)

    def gram(self, kernel, X, X_size):
        return self._gram(kernel, X, X_size)

    def cross_covariance(self, kernel, X, Z, X_size, Z_size):
        return self._cross_covariance(kernel, X, Z, X_size, Z_size)

    def diagonal(self, kernel, X, X_size):
        return self._diagonal(kernel, X, X_size)


class Signature_Kernel(gpx.kernels.AbstractKernel):
    """
    """
    def __init__(self, n_dimensions, n_timesteps, n_nontrivial_levels, lengthscales=None, weights=None, compute_engine=Custom_Compute_Engine):
        self.n_dimensions = n_dimensions
        self.n_timesteps = n_timesteps
        self.n_nontrivial_levels = n_nontrivial_levels
        self.compute_engine = compute_engine() if isinstance(compute_engine, type) else compute_engine
        self.lengthscales = NonNegativeReal(lengthscales if lengthscales is not None else self.default_lengthscales())
        self.weights = NonNegativeReal(weights if weights is not None else self.default_weights())
        super().__init__(active_dims=slice(None), n_dims=n_dimensions * n_timesteps, compute_engine=self.compute_engine)

    def default_lengthscales(self):
        C = jnp.sqrt(2*self.n_dimensions)
        return C * jnp.ones(self.n_dimensions)

    def default_weights(self):
        return jnp.ones(self.n_nontrivial_levels+1)

    def incorporate_lengthscales(self, X):
        X_scaled = X / (self.lengthscales[None, :, None])
        return X_scaled

    def gram(self, X):
        X_size = X.shape[0]
        X = jnp.reshape(X, (X_size, self.n_dimensions, self.n_timesteps))
        X_scaled = self.incorporate_lengthscales(X)
        return psd(Dense(self.compute_engine.gram(self, X_scaled, X_size)))

    def cross_covariance(self, X, Z):
        X_size = X.shape[0]
        Z_size = Z.shape[0]
        X = jnp.reshape(X, (X_size, self.n_dimensions, self.n_timesteps))
        Z = jnp.reshape(Z, (Z_size, self.n_dimensions, self.n_timesteps))
        X_scaled = self.incorporate_lengthscales(X)
        Z_scaled = self.incorporate_lengthscales(Z)
        levels = self.compute_engine.cross_covariance(self, X_scaled, Z_scaled, X_size, Z_size)
        return jnp.tensordot(self.weights, levels, axes=([0], [0]))

    def diagonal(self, X):
        X_size = X.shape[0]
        X = jnp.reshape(X, (X_size, self.n_dimensions, self.n_timesteps))
        X_scaled = self.incorporate_lengthscales(X)
        levels = self.compute_engine.diagonal(self, X_scaled, X_size)
        return jnp.tensordot(self.weights, levels, axes=([0], [0]))

    def __call__(self, X, Z):
        raise NotImplementedError("Pointwise kernel evaluation is not implemented.")
