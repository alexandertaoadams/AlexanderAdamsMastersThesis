import gpjax as gpx
from flax import nnx


def custom_fit(model, objective, train_data, optimiser, num_iters, batch_size, verbose=False):

    intermediate_model, history1 = gpx.fit(
        model=model,
        objective= lambda model, data: -objective(model, data),
        train_data=train_data,
        optim=optimiser,
        trainable=nnx.OfType((gpx.parameters.Real, gpx.parameters.LowerTriangular)),
        num_iters=num_iters,
        batch_size=batch_size,
        verbose=verbose
    )

    optimised_model, history2 = gpx.fit(
        model=intermediate_model,
        objective= lambda model, data: -objective(model, data),
        train_data=train_data,
        optim=optimiser,
        trainable=gpx.parameters.Parameter,
        num_iters=num_iters,
        batch_size=batch_size,
        verbose=verbose
    )

    return optimised_model
