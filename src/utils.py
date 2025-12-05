import numpy as np
import jax 
import jax.numpy as jnp
import pandas as pd
import sklearn.metrics as skm


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


def display_parameters(model):
  
    lengthscales = model.posterior.prior.kernel.lengthscales.value
    amplitude = model.posterior.prior.kernel.amplitude.value
    weights = model.posterior.prior.kernel.weights.value

    data = [
        ["Lengthscales", len(lengthscales), lengthscales],
        ["Amplitude", len(amplitude), amplitude],
        ["Weights", len(weights), weights],
    ]


    df = pd.DataFrame(data, columns=["Parameter", "Count", "Values"])
    return df

def display_results(pred_labels, true_labels):

    y_pred = np.array(pred_labels)
    y_true = np.array(true_labels)
    
    total = int(len(y_true))
    num_neg = int(np.sum(y_true == 0))
    num_pos = int(np.sum(y_true == 1))
    mcc = float(skm.matthews_corrcoef(y_true, y_pred))
    f1 = float(skm.f1_score(y_true, y_pred))
    precision = float(skm.precision_score(y_true, y_pred))
    recall = float(skm.recall_score(y_true, y_pred))

    data = {
        "Metric": [
            "Test Size",
            "Negative Samples",
            "Postive Samples",
            "MCC",
            "F1 score",
            "Precision",
            "Recall"
        ],
        "Value": [
            total,
            num_neg,
            num_pos,
            round(mcc, 3),
            round(f1, 3),
            round(precision, 3),
            round(recall, 3)
        ]
    }

    results_table = pd.DataFrame(data).set_index("Metric")
    return results_table
