import jax
import jax.numpy as jnp


@jax.jit
def predict_batch(model, batch):
    def predict_single(x):
        latent_dist = model.predict(x[None, :])
        predicted_dist = model.posterior.likelihood(latent_dist)
        return predicted_dist.mean.squeeze()
    return jax.vmap(predict_single)(batch)

def batched_predict(xtest, model, batch_size=100):
    num_points = xtest.shape[0]
    num_batches = (num_points + batch_size - 1) // batch_size

    results = []
    for i in range(num_batches):
        batch = xtest[i * batch_size : (i + 1) * batch_size]
        preds = predict_batch(model, batch)
        results.append(preds)

    return jnp.concatenate(results, axis=0)
