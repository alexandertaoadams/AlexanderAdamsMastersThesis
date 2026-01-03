import numpy as np

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
