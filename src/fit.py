import jax
import jax.numpy as jnp
import gpjax as gpx
import optax as ox
from gpjax.objectives import elbo
from gpjax.parameters import Parameter
from gpjax.dataset import Dataset

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
):
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

        jax.debug.print(
            "Steps: {s}, elapsed: {t:.2f}s, last loss: {l}",
            s=step_count,
            t=elapsed,
            l=losses[-1],
        )

    if params_bijection is not None:
        params = transform(params, params_bijection)

    trained_model = nnx.merge(graphdef, params, *static_state)
    return trained_model



def timed_fit_with_validation(
    model,
    objective,
    train_data,
    validation_data,
    optim,
    params_bijection: tp.Union[dict[Parameter, Transform], None] = DEFAULT_BIJECTION,
    trainable=Parameter,
    max_time=300,
    iterations_per_stage=100,
    patience=10,
    seed=42,
    batch_size=-1,
    unroll=1
):

    graphdef, params, *static_state = nnx.split(model, trainable, ...)

    if params_bijection is not None:
        params = transform(params, params_bijection, inverse=True)

    def loss(params: nnx.State, batch):
        params_c = transform(params, params_bijection)
        model = nnx.merge(graphdef, params_c, *static_state)
        return objective(model, batch)

    def step(carry, key):
        params, opt_state = carry

        if batch_size != -1:
            batch = get_batch(train_data, batch_size, key)
        else:
            batch = train_data

        loss_val, grads = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optim.update(grads, opt_state, params)
        params = ox.apply_updates(params, updates)

        return (params, opt_state), loss_val
        
    key = jr.PRNGKey(seed)
    opt_state = optim.init(params)

    best_params = params
    best_val_loss = jnp.inf
    bad_stages = 0
    stage = 0

    start = time.perf_counter()
    elapsed = 0.0
    while elapsed < max_time and bad_stages < patience:

        key, stage_key = jr.split(key)
        iter_keys = jr.split(stage_key, iterations_per_stage)

        # JAX scan 
        (new_params, new_opt_state), _ = jax.lax.scan(
            step,
            (params, opt_state),
            iter_keys,
            unroll=unroll,
        )

        # Validation
        params_c = transform(new_params, params_bijection)
        model = nnx.merge(graphdef, params_c, *static_state)
        new_val_loss = float(-elbo(model, validation_data))  
        jax.debug.print("Stage {s}, validation loss = {l}", s=stage, l=new_val_loss)

        # Update for next stage
        params = new_params
        opt_state = new_opt_state

        # Early stopping update
        if new_val_loss < best_val_loss:
            best_val_loss = new_val_loss
            best_params = new_params
            bad_stages = 0
        else:
            bad_stages += 1

        # Update elapsed and stage
        stage += 1
        elapsed = time.perf_counter() - start
        jax.debug.print("Elapsed time: {t:.2f}s", t=elapsed)

    if params_bijection is not None:
        best_params = transform(best_params, params_bijection)

    trained_model = nnx.merge(graphdef, best_params, *static_state)
    return trained_model



def convergence_fit(
    model,
    objective,
    train_data,
    validation_dataset,
    optim,
    params_bijection: tp.Union[dict[Parameter, Transform], None] = DEFAULT_BIJECTION,
    trainable=Parameter,
    seed=42,
    batch_size=-1,
    unroll=1,
    max_stages=10,
    num_iters=100,
    patience=3,
):

    graphdef, params, *static_state = nnx.split(model, trainable, ...)

    # Parameters bijection to unconstrained space
    if params_bijection is not None:
        params = transform(params, params_bijection, inverse=True)

    def loss(params: nnx.State, batch):
        params_c = transform(params, params_bijection)
        model = nnx.merge(graphdef, params_c, *static_state)
        return objective(model, batch)

    def step(carry, key):
        params, opt_state = carry

        if batch_size != -1:
            batch = get_batch(train_data, batch_size, key)
        else:
            batch = train_data

        loss_val, grads = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optim.update(grads, opt_state, params)
        params = ox.apply_updates(params, updates)

        return (params, opt_state), loss_val

    def cond_fun(state):
        params, opt_state, best_val_loss, bad_stages, stage, key = state
        return (stage < max_stages) & (bad_stages < patience)

    def body_fun(state):
        params, opt_state, best_val_loss, bad_stages, stage, key = state

        key, stage_key = jr.split(key)
        iter_keys = jr.split(stage_key, num_iters)

        (new_params, new_opt_state), _ = jax.lax.scan(
            step,
            (params, opt_state),
            iter_keys,
            unroll=unroll,
        )

        params_c = transform(new_params, params_bijection)
        model = nnx.merge(graphdef, params_c, *static_state)
        new_val_loss = -elbo(model, validation_dataset)
        jax.debug.print("Stage {s}, validation loss = {l}", s=stage, l=new_val_loss)

        improved = new_val_loss < best_val_loss

        best_val_loss = jnp.where(improved, new_val_loss, best_val_loss)
        bad_stages = jnp.where(improved, 0, bad_stages + 1)

        return (
            new_params,
            new_opt_state,
            best_val_loss,
            bad_stages,
            stage + 1,
            key,
        )

    key = jr.PRNGKey(seed)
    opt_state = optim.init(params)

    best_params = params
    best_val_loss = jnp.inf
    bad_stages = 0
    stage = 0

    while stage < max_stages and bad_stages < patience:
        (
            params,
            opt_state,
            best_val_loss_new,
            bad_stages,
            stage,
            key,
        ) = jax.lax.while_loop(
            cond_fun,
            body_fun,
            (params, opt_state, best_val_loss, bad_stages, stage, key),
        )

        if best_val_loss_new < best_val_loss:
            best_val_loss = best_val_loss_new
            best_params = params

    if params_bijection is not None:
        best_params = transform(best_params, params_bijection)

    trained_model = nnx.merge(graphdef, best_params, *static_state)
    return trained_model




def get_batch(train_data: Dataset, batch_size: int, key) -> Dataset:
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

    return Dataset(X=x[indices], y=y[indices])
