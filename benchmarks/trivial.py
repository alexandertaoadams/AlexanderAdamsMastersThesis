



class TrivialCustomComputeEngine(gpx.kernels.computations.AbstractKernelComputation):

    def _gram(self, kernel, X, X_size):
        lengthscales = kernel.lengthscales

        weights = kernel.weights
        return Gram_XX_Trivial_jit(
            X, X_size, kernel.n_timesteps, kernel.n_nontrivial_levels, lengthscales.get_value(), weights.get_value()
            )

    def _cross_covariance(self, kernel, X, Z, X_size, Z_size):
        lengthscales = kernel.lengthscales

        weights = kernel.weights
        return Cross_XZ_Trivial_jit(
            X, Z, X_size, Z_size, kernel.n_timesteps, kernel.n_nontrivial_levels, lengthscales.get_value(), weights.get_value()
            )

    def _diagonal(self, kernel, X, X_size):
        lengthscales = kernel.lengthscales

        weights = kernel.weights
        return Diag_XX_Trivial_jit(
            X, X_size, kernel.n_timesteps, kernel.n_nontrivial_levels, lengthscales.get_value(), weights.get_value()
            )

    def gram(self, kernel, X, X_size):
        return self._gram(kernel, X, X_size)

    def cross_covariance(self, kernel, X, Z, X_size, Z_size):
        return self._cross_covariance(kernel, X, Z, X_size, Z_size)

    def diagonal(self, kernel, X, X_size):
        return self._diagonal(kernel, X, X_size)



class TrivialSignatureKernel(gpx.kernels.AbstractKernel):

    def __init__(self, n_dimensions, n_timesteps, n_nontrivial_levels, lengthscales=None, weights=None, compute_engine=TrivialCustomComputeEngine):
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

    def default_amplitude(self):
        return jnp.array([1.0])

    def default_weights(self):
        return jnp.ones(self.n_nontrivial_levels+1)

    def reshape3D(self, X):
        return jnp.reshape(X, (X.shape[0], self.n_dimensions, self.n_timesteps))

    def gram(self, X):
        X_size = X.shape[0]
        X = self.reshape3D(X)
        return psd(Dense(self.compute_engine.gram(self, X, X_size)))

    def cross_covariance(self, X, Z):
        X_size = X.shape[0]
        Z_size = Z.shape[0]
        X = self.reshape3D(X)
        Z = self.reshape3D(Z)
        return self.compute_engine.cross_covariance(self, X, Z, X_size, Z_size)

    def diagonal(self, X):
        X_size = X.shape[0]
        X = self.reshape3D(X)
        return self.compute_engine.diagonal(self, X, X_size)

    def __call__(self, X, Z):
        raise NotImplementedError("Pointwise kernel evaluation is not implemented.")



def Gram_XX_Trivial(X, X_batch_size, n_timesteps, n_nontrivial_levels, lengthscales, weights):

    n_X = X_batch_size
    T = n_timesteps

    def incorporate_lengthscales(X, lengthscales):
        X_scaled = X / (lengthscales[None, :, None])
        return X_scaled

    X = incorporate_lengthscales(X, lengthscales)

    XX = jax.lax.dot_general(X, X, dimension_numbers=( ((1,),(1,)), ((),()) ) )
    S = jnp.transpose(XX, (0, 2, 1, 3))
    R = S[:,:,1:,1:] - S[:,:,1:,:-1] - S[:,:,:-1,1:] + S[:,:,:-1,:-1]

    # Level 0 (Trivial Level)
    level_0 = jnp.expand_dims(jnp.ones((n_X, n_X)), axis=0)

    # Level 1
    level_1 = jnp.expand_dims(jnp.sum(R, axis=(-1, -2)), axis=0)

    # Prefactors matrix needed for levels > 1
    multipliers = jnp.arange(1, n_nontrivial_levels+2)
    P_reciprocal = jnp.outer(multipliers, multipliers)
    P = 1./P_reciprocal
    P = P[..., None, None, None, None]

    # Levels > 1
    def scan_function(C_prev, idx):

        # Element (0, 0)
        C_00 = R * jnp.cumulative_sum(jnp.cumulative_sum(jnp.sum(C_prev, axis=(0,1)), axis=-1, include_initial=True), axis=-2, include_initial=True)[:,:,:-1,:-1]

        # Elements(idx>=r>0, idx>=s>0)
        C_inner = P[1:,1:] * R * C_prev

        # Elements (0, ) / first row
        C_row = P[0, 1:] * R * jnp.cumulative_sum(jnp.sum(C_prev, axis=0), axis=-2, include_initial=True)[:,:,:,:-1,:]

        # Elements (, 0) / first column
        C_col = P[1:, 0] * R * jnp.cumulative_sum(jnp.sum(C_prev, axis=1), axis=-1, include_initial=True)[:,:,:,:,:-1]

        # Combining elements
        C_00 = jnp.expand_dims(jnp.expand_dims(C_00, axis=0), axis=0)
        C_row = jnp.expand_dims(C_row, axis=0)
        C_col = jnp.expand_dims(C_col, axis=1)
        C_1 = jnp.concatenate((C_00, C_row), axis=1)
        C_2 = jnp.concatenate((C_col, C_inner), axis=1)
        C = (jnp.concatenate((C_1, C_2), axis=0))[:-1,:-1]
        level = jnp.sum(C, axis=(0,1,-1,-2))
        return C, level

    # Initialise the carry and iterate
    C_initial = jnp.zeros((n_nontrivial_levels, n_nontrivial_levels) + R.shape)
    C_initial = C_initial.at[0,0].set(R)
    C_final, higher_levels = jax.lax.scan(scan_function, C_initial, xs=jnp.arange(2, n_nontrivial_levels+1))

    L = jnp.concat([level_0, level_1, higher_levels], axis=0)
    return jnp.tensordot(weights, L , axes=([0], [0]))

Gram_XX_Trivial_jit = jax.jit(Gram_XX_Trivial, static_argnames=['X_batch_size', 'n_timesteps', 'n_nontrivial_levels'])

def Cross_XZ_Trivial(X, Z, X_batch_size, Z_batch_size, n_timesteps, n_nontrivial_levels, lengthscales, weights):

    n_X = X_batch_size
    n_Z = Z_batch_size
    T = n_timesteps

    def incorporate_lengthscales(X, lengthscales):
        X_scaled = X / (lengthscales[None, :, None])
        return X_scaled

    X = incorporate_lengthscales(X, lengthscales)
    Z = incorporate_lengthscales(Z, lengthscales)

    XZ = jax.lax.dot_general(X, Z, dimension_numbers=( ((1,),(1,)), ((),()) ) )
    S = jnp.transpose(XZ, (0, 2, 1, 3))
    R = S[:,:,1:,1:] - S[:,:,1:,:-1] - S[:,:,:-1,1:] + S[:,:,:-1,:-1]

    # Level 0 (Trivial Level)
    level_0 = jnp.expand_dims(jnp.ones((n_X, n_Z)), axis=0)

    # Level 1
    level_1 = jnp.expand_dims(jnp.sum(R, axis=(-1, -2)), axis=0)

    # Prefactors matrix needed for levels > 1
    multipliers = jnp.arange(1, n_nontrivial_levels+2)
    P_reciprocal = jnp.outer(multipliers, multipliers)
    P = 1./P_reciprocal
    P = P[..., None, None, None, None]

    # Levels > 1
    def scan_function(C_prev, idx):

        # Element (0,0)
        C_00 = R * jnp.cumulative_sum(jnp.cumulative_sum(jnp.sum(C_prev, axis=(0,1)), axis=-1, include_initial=True), axis=-2, include_initial=True)[:,:,:-1,:-1]

        # Elements(idx>=r>0, idx>=s>0)
        C_inner = P[1:,1:] * R * C_prev

        # Elements (0, ) / first row
        C_row = P[0, 1:] * R * jnp.cumulative_sum(jnp.sum(C_prev, axis=0), axis=-2, include_initial=True)[:,:,:,:-1,:]

        # Elements (, 0) / first column
        C_col = P[1:, 0] * R * jnp.cumulative_sum(jnp.sum(C_prev, axis=1), axis=-1, include_initial=True)[:,:,:,:,:-1]

        # Combining elements
        C_00 = jnp.expand_dims(jnp.expand_dims(C_00, axis=0), axis=0)
        C_row = jnp.expand_dims(C_row, axis=0)
        C_col = jnp.expand_dims(C_col, axis=1)
        C_1 = jnp.concatenate((C_00, C_row), axis=1)
        C_2 = jnp.concatenate((C_col, C_inner), axis=1)
        C = (jnp.concatenate((C_1, C_2), axis=0))[:-1,:-1]
        level = jnp.sum(C, axis=(0,1,-1,-2))
        return C, level

    # Initialise the carry and iterate
    C_initial = jnp.zeros((n_nontrivial_levels, n_nontrivial_levels) + R.shape)
    C_initial = C_initial.at[0,0].set(R)
    C_final, higher_levels = jax.lax.scan(scan_function, C_initial, xs=jnp.arange(2, n_nontrivial_levels+1))

    L = jnp.concat([level_0, level_1, higher_levels], axis=0)
    return jnp.tensordot(weights, L , axes=([0], [0]))

Cross_XZ_Trivial_jit = jax.jit(Cross_XZ_Trivial, static_argnames=['X_batch_size', 'Z_batch_size', 'n_timesteps', 'n_nontrivial_levels'])

def Diag_XX_Trivial(X, X_batch_size, n_timesteps, n_nontrivial_levels, lengthscales, weights):
    n_X = X_batch_size
    T = n_timesteps

    def incorporate_lengthscales(X, lengthscales):
        X_scaled = X / (lengthscales[None, :, None])
        return X_scaled

    X = incorporate_lengthscales(X, lengthscales)

    # Static lifting of data points with the radial basis function
    XX = jax.lax.dot_general(X, X, dimension_numbers=( ((1,),(1,)), ((0,),(0,)) ) )
    S = XX
    R = S[:,1:,1:] - S[:,1:,:-1] - S[:,:-1,1:] + S[:,:-1,:-1]

    # Level 0 (Trivial Level)
    level_0 = jnp.expand_dims(jnp.ones((n_X)), axis=0)

    # Level 1
    level_1 = jnp.expand_dims(jnp.sum(R, axis=(-1, -2)), axis=0)

    # Prefactors matrix
    multipliers = jnp.arange(1, n_nontrivial_levels+2)
    P_reciprocal = jnp.outer(multipliers, multipliers)
    P = 1./P_reciprocal
    P = P[..., None, None, None]

    # Levels > 1
    def scan_function(C_prev, idx):

        # Element (0,0)
        C_00 = R * jnp.cumulative_sum(jnp.cumulative_sum(jnp.sum(C_prev, axis=(0,1)), axis=-1, include_initial=True), axis=-2, include_initial=True)[:,:-1,:-1]

        # Elements(idx>=r>0, idx>=s>0)
        C_inner = P[1:,1:] * R * C_prev

        # Elements (0, ) / first row
        C_row = P[0, 1:] * R * jnp.cumulative_sum(jnp.sum(C_prev, axis=0), axis=-2, include_initial=True)[:,:,:-1,:]

        # Elements (, 0) / first column
        C_col = P[1:, 0] * R * jnp.cumulative_sum(jnp.sum(C_prev, axis=1), axis=-1, include_initial=True)[:,:,:,:-1]

        # Combining elements
        C_00 = jnp.expand_dims(jnp.expand_dims(C_00, axis=0), axis=0)
        C_row = jnp.expand_dims(C_row, axis=0)
        C_col = jnp.expand_dims(C_col, axis=1)
        C_1 = jnp.concatenate((C_00, C_row), axis=1)
        C_2 = jnp.concatenate((C_col, C_inner), axis=1)
        C = (jnp.concatenate((C_1, C_2), axis=0))[:-1,:-1]
        level = jnp.sum(C, axis=(0,1,-1,-2))
        return C, level

    # Initialise the carry and iterate
    C_initial = jnp.zeros((n_nontrivial_levels, n_nontrivial_levels) + R.shape)
    C_initial = C_initial.at[0,0].set(R)
    C_final, higher_levels = jax.lax.scan(scan_function, C_initial, xs=jnp.arange(2, n_nontrivial_levels+1))
    L = jnp.concat([level_0, level_1, higher_levels], axis=0)
    return jnp.tensordot(weights, L , axes=([0], [0]))

Diag_XX_Trivial_jit = jax.jit(Diag_XX_Trivial, static_argnames=['X_batch_size', 'n_timesteps', 'n_nontrivial_levels'])
