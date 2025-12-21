import jax
import jax.numpy as jnp
import gpjax as gpx
from jax import vmap
from gpjax.linalg import Dense, psd
from gpjax.linalg.utils import add_jitter
from gpjax.linalg.operations import lower_cholesky, solve
from gpjax.distributions import GaussianDistribution
from gpjax.distributions import _kl_divergence as kl_divergence


def collapsed_elbo_bernoulli(variational_family, data):

    # Notation and derivation:
    #
    # Let Sigma = (K_{mm} + \sigma^{-2} K_{mx} K_{xm})^{-1}.
    # Then the optimal distribution for inducing variables f_m is p(f_m) = N(f_m|mu, A), where:
    #     mu = o
    #
    #
    # Let Q = KxzKzz⁻¹Kzx, we must compute the log normal pdf:
    #
    #   log N(y; μx, o²I + Q) = -nπ - n/2 log|o²I + Q|
    #   - 1/2 (y - μx)ᵀ (o²I + Q)⁻¹ (y - μx).
    #
    # The log determinant |o²I + Q| is computed via applying the matrix determinant
    #   lemma
    #
    #   |o²I + Q| = log|o²I| + log|I + Lz⁻¹ Kzx (o²I)⁻¹ Kxz Lz⁻¹| = log(o²) +  log|B|,
    #
    #   with B = I + AAᵀ and A = Lz⁻¹ Kzx / o.
    #
    # Similarly we apply matrix inversion lemma to invert o²I + Q
    #
    #   (o²I + Q)⁻¹ = (Io²)⁻¹ - (Io²)⁻¹ Kxz Lz⁻ᵀ (I + Lz⁻¹ Kzx (Io²)⁻¹ Kxz Lz⁻ᵀ )⁻¹ Lz⁻¹ Kzx (Io²)⁻¹
    #               = (Io²)⁻¹ - (Io²)⁻¹ oAᵀ (I + oA (Io²)⁻¹ oAᵀ)⁻¹ oA (Io²)⁻¹
    #               = I/o² - Aᵀ B⁻¹ A/o²,
    #
    # giving the quadratic term as
    #
    #   (y - μx)ᵀ (o²I + Q)⁻¹ (y - μx) = [(y - μx)ᵀ(y - µx)  - (y - μx)ᵀ Aᵀ B⁻¹ A (y - μx)]/o²,
    #
    #   with A and B defined as above.
    
    # Unpack training data
    x, y, n = data.X, data.y, data.n
    z = variational_family.inducing_inputs
    m = variational_family.num_inducing

    # Unpack mean function and kernel
    mean_function = variational_family.posterior.prior.mean_function
    kernel = variational_family.posterior.prior.kernel

    # 
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
    p_inducing = GaussianDistribution(jnp.atleast_1d(mean_function(z).squeeze()), scale=psd(Dense(Kmm)))
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

    # Compute Mu_tilde
    M4 = solve(L1, Kmn)
    D = jnp.matmul(M4.T, M4)
    mu_tilde = jnp.matmul(D, y - mean_x)

    # QH quadrature to approximate the integrals, using A_tilde and Mu_tilde
    link_func = variational_family.posterior.likelihood.link_function
    log_prob = vmap(lambda f, y: link_func(f).log_prob(y))
    integrals = variational_family.posterior.likelihood.integrator(fun=log_prob, y=y, mean=mu_tilde, variance=A_tilde, likelihood=variational_family.posterior.likelihood)

    # Lower bound on ELBO is given by the following:
    elbolbo = jnp.sum(integrals) - kl
    return elbolbo
