import numpy as np
import jax 
import jax.numpy as jnp
import pandas as pd
import sklearn.metrics as skm

def display_parameters(model):
    """Print a table containing the model hyperparameters: lengthscales, amplitude, weights.
    """
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

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    mcc = float(skm.matthews_corrcoef(y_true, y_pred))
    f1 = float(skm.f1_score(y_true, y_pred))

    data = {
        "Metric": [
            "Test Size",
            "Negative Samples",
            "Positive Samples",
            "True Positives (TP)",
            "True Negatives (TN)",
            "False Positives (FP)",
            "False Negatives (FN)",
            "MCC",
            "F1 score",
        ],
        "Value": [
            total,
            num_neg,
            num_pos,
            tp,
            tn,
            fp,
            fn,
            round(mcc, 3),
            round(f1, 3),
        ]
    }

    results_table = pd.DataFrame(data).set_index("Metric")
    return results_table

    results_table = pd.DataFrame(data).set_index("Metric")
    return results_table
