import jax
import jax.numpy as jnp
import gpjax as gpx
from jax import jit


def Gram_XX(X, X_batch_size, n_timesteps, n_nontrivial_levels, lengthscales, amp, weights):
    """Computes the gram matrix on the input data and returns the levels separately.
    Args:
        X (X_batch_size, n_dimensions, n_timesteps) = (n_X, D, T): The time series data which we want to compute the gram matrix of.
        X_batch_size (n_X): The number of data points.
        n_timesteps (T): The length of each time series datum.
        n_nontrivial_levels: The number of nontrivial signature levels we want to compute.
        lengthscales (D): The lengthscales parameter
        amp (1): The amplitude parameter
        weights (n_nontrivial levels + 1): The weights parameter

    Intermediates:
        S (n_X, n_X, T, T): Matrix containing squared euclidian distances between pairs of points in X.
        RBF (n_X, n_X, T, T): Radial basis function applied to S.
        R (n_X, n_X, T-1, T-1): Delta of RBF.
        C (n_nontrivial_levels, n_nontrivial_levels, n_X, n_X, T-1, T-1): The carry of lax.scan.

    Returns:
        Gram matrix (n_X, n_X): The weighted sum of the signature levels.
    """
    n_X = X_batch_size
    T = n_timesteps

    def incorporate_lengthscales(X, lengthscales):
        X_scaled = X / (lengthscales[None, :, None])
        return X_scaled

    X = incorporate_lengthscales(X, lengthscales)

    # Static lifting of data points with the radial basis function
    X__2 = jnp.sum(X**2, axis=1)
    X__2X__2 = jnp.add.outer(X__2, X__2)
    XX = jax.lax.dot_general(X, X, dimension_numbers=( ((1,),(1,)), ((),()) ) )
    S = jnp.transpose(X__2X__2  - 2*XX, (0, 2, 1, 3))
    RBF = amp * jnp.exp(-0.5 * S)
    R = RBF[:,:,1:,1:] - RBF[:,:,1:,:-1] - RBF[:,:,:-1,1:] + RBF[:,:,:-1,:-1]

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
    X_var = diagonal_jit(X, X_batch_size, n_timesteps, n_nontrivial_levels, lengthscales, amp, weights) 
    L = L / (jnp.sqrt(X_var)[:, :, None] * jnp.sqrt(X_var)[:, None, :])

    return jnp.tensordot(weights, L , axes=([0], [0]))

Gram_XX_jit = jax.jit(Gram_XX, static_argnames=['X_batch_size', 'n_timesteps', 'n_nontrivial_levels'])

def Cross_XZ(X, Z, X_batch_size, Z_batch_size, n_timesteps, n_nontrivial_levels, lengthscales, amp, weights):
    """Computes the cross covariance matrix between the input data X and Z and returns the levels separately.
    Args:
        X (X_batch_size, n_dimensions, n_timesteps) = (n_X, D, T): Time series data 1.
        Z (Z_batch_size, n_dimensions, n_timesteps) = (n_Z, D, T): Time series data 2.
        X_batch_size (n_X): The number of data points in X.
        Z_batch_size (n_Z): The number of data points in Z.
        n_timesteps (T): The length of each time series datum, whether in X or Z.
        n_nontrivial_levels: The number of nontrivial signature levels we want to compute.
        lengthscales (D): The lengthscales parameter
        amp (1): The amplitude parameter
        weights (n_nontrivial levels + 1): The weights parameter

    Intermediates:
        S (n_X, n_Z, T, T): Matrix containing squared euclidian distances between pairs of points in X and Z.
        RBF (n_X, n_Z, T, T): Radial basis function applied to S.
        R (n_X, n_Z, T-1, T-1): Delta of RBF.
        C (n_nontrivial_levels, n_nontrivial_levels, n_X, n_Z, T-1, T-1): The carry of lax.scan.

    Returns:
        Cross covariance matrix (n_X, n_Z): The weighted sum of the signature levels.
    """
    n_X = X_batch_size
    n_Z = Z_batch_size
    T = n_timesteps

    def incorporate_lengthscales(X, lengthscales):
        X_scaled = X / (lengthscales[None, :, None])
        return X_scaled

    X = incorporate_lengthscales(X, lengthscales)
    Z = incorporate_lengthscales(Z, lengthscales)

    # Static lifting of data points with the radial basis function
    X__2 = jnp.sum(X**2, axis=1)
    Z__2 = jnp.sum(Z**2, axis=1)
    X__2Z__2 = jnp.add.outer(X__2, Z__2)
    XZ = jax.lax.dot_general(X, Z, dimension_numbers=( ((1,),(1,)), ((),()) ) )
    S = jnp.transpose(X__2Z__2 - 2*XZ, (0, 2, 1, 3))
    RBF = amp * jnp.exp(-0.5 * S)
    R = RBF[:,:,1:,1:] - RBF[:,:,1:,:-1] - RBF[:,:,:-1,1:] + RBF[:,:,:-1,:-1]

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
    X_var = diagonal_jit(X, X_batch_size, n_timesteps, n_nontrivial_levels, lengthscales, amp, weights)
    Z_var = diagonal_jit(Z, Z_batch_size, n_timesteps, n_nontrivial_levels, lengthscales, amp, weights)
    X_std = jnp.sqrt(X_var)  
    Z_std = jnp.sqrt(Z_var)  
    L = L / X_std[..., :, None]   
    L = L / Z_std[..., None, :] 

    return jnp.tensordot(weights, L, axes=([0], [0]))
    

Cross_XZ_jit = jax.jit(Cross_XZ, static_argnames=['X_batch_size', 'Z_batch_size', 'n_timesteps', 'n_nontrivial_levels'])



def diagonal(X, X_batch_size, n_timesteps, n_nontrivial_levels, lengthscales, amp, weights):
    """Computes just the diagonal elements of the Gram matrix of the input data and returns the levels separately.
    Args:
        X (X_batch_size, n_dimensions, n_timesteps) = (n_X, D, T): Time series data.
        X_batch_size (n_X): The number of data points in X.
        n_timesteps (T): The length of each time series datum.
        n_nontrivial_levels: The number of nontrivial signature levels we want to compute.
        lengthscales (D): The lengthscales parameter
        amp (1): The amplitude parameter
        weights (n_nontrivial levels + 1): The weights parameter

    Intermediates:
        S (n_X, T, T): For each input datum we compute the squared Euclidean distance between every pair of timestaps and store in S. We do this for each datum.
        RBF (n_X, T, T): Radial basis function applied to S.
        R (n_X, T-1, T-1): Delta of RBF.
        C (n_nontrivial_levels, n_nontrivial_levels, n_X, T-1, T-1): The carry of lax.scan.

    Returns:
        Diagonal entries of the Gram matrix (n_X): The weighted sum of the signature levels.
    """
    n_X = X_batch_size
    T = n_timesteps

    def incorporate_lengthscales(X, lengthscales):
        X_scaled = X / (lengthscales[None, :, None])
        return X_scaled

    X = incorporate_lengthscales(X, lengthscales)

    # Static lifting of data points with the radial basis function
    X__2_d = jnp.sum(X**2, axis=1)
    X__2X__2_d = X__2_d[:, :, None] + X__2_d[:, None, :]
    XX = jax.lax.dot_general(X, X, dimension_numbers=( ((1,),(1,)), ((0,),(0,)) ) )
    S = X__2X__2_d  - 2*XX
    RBF = amp*jnp.exp(-0.5 * S)
    R = RBF[:,1:,1:] - RBF[:,1:,:-1] - RBF[:,:-1,1:] + RBF[:,:-1,:-1]

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

    return L 


diagonal_jit = jax.jit(diagonal, static_argnames=['X_batch_size', 'n_timesteps', 'n_nontrivial_levels'])
