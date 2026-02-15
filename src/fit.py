import jax
import jax.numpy as jnp
import gpjax as gpx
import optax as ox
from gpjax.objectives import elbo
from gpjax.parameters import Parameter
from .datasets import Dataset_3D

from flax import nnx
import time 
import typing as tp
import jax.random as jr
from gpjax.parameters import (
    DEFAULT_BIJECTION,
    Parameter,
    transform,
)
from numpyro.distributions.transforms import Transform



def iteration_fit(
    model,
    objective,
    train_data,
    optim,
    params_bijection: dict[Parameter, Transform] | None = DEFAULT_BIJECTION,
    trainable: nnx.filterlib.Filter = Parameter,
    key = jr.key(42),
    num_iters: int = 100,
    batch_size: int = -1,
    log_rate: int = 10,
    verbose: bool = True,
    unroll: int = 1,
    safe: bool = True,
):

    if safe:
        _check_model(model)
        _check_optim(optim)
        _check_batch_size(batch_size)

    # Model state filtering
    graphdef, params, *static_state = nnx.split(model, trainable, ...)

    # Parameters bijection to unconstrained space
    if params_bijection is not None:
        params = transform(params, params_bijection, inverse=True)

    # Loss definition
    def loss(params, batch):
        params = transform(params, params_bijection)
        model = nnx.merge(graphdef, params, *static_state)
        return objective(model, batch)

    # Initialise optimiser state.
    opt_state = optim.init(params)

    # Mini-batch random keys to scan over.
    iter_keys = jr.split(key, num_iters)

    # Optimisation step.
    def step(carry, key):
        params, opt_state = carry

        if batch_size != -1:
            batch = get_batch(train_data, batch_size, key)
        else:
            batch = train_data

        loss_val, loss_gradient = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optim.update(loss_gradient, opt_state, params)
        params = ox.apply_updates(params, updates)

        carry = params, opt_state
        return carry, loss_val

    # Optimisation scan.
    scan = vscan if verbose else jax.lax.scan

    # Optimisation loop.
    (params, _), history = scan(step, (params, opt_state), (iter_keys), unroll=unroll)

    # Parameters bijection to constrained space
    if params_bijection is not None:
        params = transform(params, params_bijection)

    # Reconstruct model
    model = nnx.merge(graphdef, params, *static_state)

    return model, history



def timed_fit(
    model,
    objective,
    train_data,
    optim,
    params_bijection: tp.Union[dict[Parameter, Transform], None] = DEFAULT_BIJECTION,
    trainable=Parameter,
    max_time=300.0,          
    check_every=100,         
    seed=42,
    batch_size=-1,
    unroll=1,
    safe = True,
):

    if safe:
        _check_model(model)
        _check_optim(optim)
        _check_batch_size(batch_size)
        
    graphdef, params, *static_state = nnx.split(model, trainable, ...)

    if params_bijection is not None:
        params = transform(params, params_bijection, inverse=True)

    def loss_fn(params, batch):
        params_c = transform(params, params_bijection)
        model = nnx.merge(graphdef, params_c, *static_state)
        return objective(model, batch)

    def step(carry, key):
        params, opt_state = carry

        if batch_size != -1:
            batch = get_batch(train_data, batch_size, key)
        else:
            batch = train_data

        loss_val, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optim.update(grads, opt_state, params)
        params = ox.apply_updates(params, updates)

        return (params, opt_state), loss_val

    key = jr.PRNGKey(seed)
    opt_state = optim.init(params)

    start_time = time.perf_counter()
    elapsed = 0.0
    step_count = 0

    while elapsed < max_time:
        key, chunk_key = jr.split(key)
        keys = jr.split(chunk_key, check_every)

        (params, opt_state), losses = jax.lax.scan(
            step,
            (params, opt_state),
            keys,
            unroll=unroll,
        )

        step_count += check_every
        elapsed = time.perf_counter() - start_time

    if params_bijection is not None:
        params = transform(params, params_bijection)

    trained_model = nnx.merge(graphdef, params, *static_state)
    return trained_model



def get_batch(train_data, batch_size, key):
    """Batch the data into mini-batches. Sampling is done with replacement.

    Args:
        train_data (Dataset): The training dataset.
        batch_size (int): The batch size.
        key (KeyArray): The random key to use for the batch selection.

    Returns
    -------
        Dataset: The batched dataset.
    """
    x, y, n = train_data.X, train_data.y, train_data.n

    # Subsample mini-batch indices with replacement.
    indices = jr.choice(key, n, (batch_size,), replace=True)

    return Dataset_3D(X=x[indices], y=y[indices])

def _check_model(model: tp.Any) -> None:
    """Check that the model is a subclass of nnx.Module."""
    if not isinstance(model, nnx.Module):
        raise TypeError(
            "Expected model to be a subclass of nnx.Module. "
            f"Got {model} of type {type(model)}."
        )



def _check_optim(optim: tp.Any) -> None:
    """Check that the optimiser is of type GradientTransformation."""
    if not isinstance(optim, ox.GradientTransformation):
        raise TypeError(
            "Expected optim to be of type optax.GradientTransformation. "
            f"Got {optim} of type {type(optim)}."
        )


def _check_batch_size(batch_size: tp.Any) -> None:
    """Check that the batch size is of type int and positive if not minus 1."""
    if not isinstance(batch_size, int):
        raise TypeError(
            "Expected batch_size to be of type int. "
            f"Got {batch_size} of type {type(batch_size)}."
        )

    if not batch_size == -1 and not batch_size > 0:
        raise ValueError(f"Expected batch_size to be positive or -1. Got {batch_size}.")
