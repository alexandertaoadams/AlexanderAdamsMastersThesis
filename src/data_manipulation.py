import numpy as np
import jax
import jax.numpy as jnp

def normalise(data, eps=1e-12):
    """Normalises time series data so that each dimension has global zero mean and unit variance,
    across all samples and time steps.

    Args:
        data (np.ndarray): Array of shape (n_sequences, n_dims, seq_length)
        eps (float): Small constant to avoid division by zero

    Returns:
        np.ndarray: Normalised data 
    """
    mean = data.mean(axis=(0, 2), keepdims=True)  
    std = data.std(axis=(0, 2), keepdims=True)    
    normalised_data = (data - mean) / (std + eps)
    return normalised_data, mean, std

def add_time(sequences):
    length = sequences.shape[-1]
    time = np.arange(length) / (length - 1)
    time_broadcasted = np.broadcast_to(time.reshape(1, 1, length), (sequences.shape[0], 1, length))
    return np.concatenate([time_broadcasted, sequences], axis=1)

def train_validation_split(sequences, labels):

    sequences_zero = sequences[labels == 0]
    n_class0 = sequences_zero.shape[0]
    d0 = int(jnp.floor(0.8 * n_class0))
    key = jax.random.key(0)
    X0_indices = jax.random.choice(key, n_class0, shape=(d0,), replace=False)
    V0_indices = jnp.setdiff1d(jnp.arange(0, n_class0), X0_indices)
    X0 = sequences[X0_indices]  
    V0 = sequences[V0_indices]

    sequences_one = sequences[labels == 1]
    n_class1 = sequences_one.shape[0]
    d1 = int(jnp.floor(0.8 * n_class1))
    X1_indices = jax.random.choice(key, n_class1, shape=(d1,), replace=False)
    V1_indices = jnp.setdiff1d(jnp.arange(0, n_class1), X1_indices)
    X1 = sequences[X1_indices]  
    V1 = sequences[V1_indices]

    xtrain = jnp.concatenate([X0, X1], axis=0)
    ytrain = jnp.concatenate([jnp.zeros(X0.shape[0]), jnp.ones(X1.shape[0])])

    xvalidate = jnp.concatenate([V0, V1], axis=0)
    yvalidate = jnp.concatenate([jnp.zeros(V0.shape[0]), jnp.ones(V1.shape[0])])

    return xtrain, ytrain, xvalidate, yvalidate
