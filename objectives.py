import jax
import jax.numpy as jnp
import gpjax as gpx
from gpjax.linalg import Dense, psd
from gpjax.linalg.utils import add_jitter
from gpjax.linalg.operations import lower_cholesky, solve




def collapsed_elbo_bernoulli(variational_family, data):
    # Unpack training data
    x, y, n = data.X, data.y, data.n
    z = variational_family.inducing_inputs
    m = variational_family.num_inducing

    # Unpack mean function and kernel
    mean_function = variational_family.posterior.prior.mean_function
    kernel = variational_family.posterior.prior.kernel

    Kmm = add_jitter(kernel.gram(z).to_dense(), variational_family.jitter)
    Kmn = kernel.cross_covariance(z, x)
    Knn_diag = kernel.diagonal(x)

    # Mean
    mean_x = mean_function(x)

    # Compute A
    Sigma_dense = Kmm + jnp.matmul(Kmn, Kmn.T)
    Sigma_dense = add_jitter(Sigma_dense, variational_family.jitter)
    Sigma = psd(Dense(Sigma_dense))
    L1 = lower_cholesky(Sigma)
    M1 = solve(L1, Kmm)
    A = jnp.matmul(M1.T, M1)

    # Compute Mu
    front = solve(L1, Kmm)
    back = solve(L1, Kmn)
    almost = jnp.matmul(front.T, back)
    mu = jnp.matmul(almost, y - mean_x)

    # Compute the kl divergence, using A and Mu
    q_inducing = GaussianDistribution(jnp.atleast_1d(mu.squeeze()), scale=psd(Dense(A)))
    p_inducing = GaussianDistribution(jnp.atleast_1d(mean_function(Z).squeeze()), scale=psd(Dense(Kmm)))
    kl = q_inducing.kl_divergence(p_inducing)

    # Compute A_tilde
    Temp2_dense = Kmm - A
    Temp2_dense = add_jitter(Temp2_dense, variational_family.jitter)
    Temp2 = psd(Dense(Temp2_dense))
    L2 = lower_cholesky(Temp2)
    M2 = solve(L2, Kmm)
    B = jnp.matmul(M2.T, M2)
    Temp3_dense = B
    Temp3_dense = add_jitter(Temp3_dense, variational_family.jitter)
    Temp3 = psd(Dense(Temp3_dense))
    L3 = lower_cholesky(Temp3)
    M3 = solve(L3, Kmn)
    C = jnp.matmul(M3.T, M3)
    A_tilde = jnp.expand_dims(Knn_diag - jnp.diagonal(C), axis=1)

    # print(jnp.linalg.eigvalsh(Kmm))
    # print(jnp.linalg.eigvalsh(Sigma_dense))
    # print(jnp.linalg.eigvalsh(Temp2_dense))
    # print(jnp.linalg.eigvalsh(Temp3_dense))

    # Compute Mu_tilde
    M4 = solve(L1, Kmn)
    D = jnp.matmul(M4.T, M4)
    mu_tilde = jnp.matmul(D, y - mean_x)

    # QH quadrature to approximate the integrals
    link_func = variational_family.posterior.likelihood.link_function
    log_prob = vmap(lambda f, y: link_func(f).log_prob(y))
    integrals = variational_family.posterior.likelihood.integrator(fun=log_prob, y=y, mean=mu_tilde, variance=A_tilde, likelihood=variational_family.posterior.likelihood)

    # print(mu_tilde)
    # print(A_tilde)
    # print(integrals)
    elbolbo = jnp.sum(integrals) - kl
    return elbolbo
