def sample_inducing_sequences(sequences, num_inducing):
    '''Randomly samples inducing sequences from the input.

    Args:
        sequences (np.ndarray): Input array of shape (n_sequences, n_dimensions, seq_length)
        num_inducing (int): Number of sequences to sample

    Returns:
        Z: Sampled inducing sequences of shape (n_inducing, n_dimensions, seq_length)
    '''
    key = jax.random.key(0)
    N = sequences.shape[0]
    indices = jax.random.choice(key, N, shape=(num_inducing,), replace=False)
    Z = sequences[indices]  
    return Z

def initial_inducing_variables(sequences, labels, n_variables):
    '''
    Args:
        sequences (np.ndarray): Input array of shape (n_sequences, n_dimensions, seq_length)
        labels (np.ndarray): Binary class labels, shape (n_sequences, )
        n_variables (int): Total number of inducing variables to select

    Returns:
        Z: Inducing sequences of shape (n_inducing, n_dimensions, seq_length)
    '''

    if labels.ndim == 2:
        labels = jnp.squeeze(labels, axis=1)

    n_class0 = jnp.sum(labels == 0)
    n_class1 = jnp.sum(labels == 1)
    total = n_class0 + n_class1

    n0 = int(jnp.floor(n_class0 / total * n_variables))
    n1 = n_variables - n0

    Z0 = sample_inducing_sequences(sequences[labels == 0], n0)
    Z1 = sample_inducing_sequences(sequences[labels == 1], n1)

    Z = jnp.concatenate([Z0, Z1], axis=0)

    return Z
