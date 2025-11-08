import jax
import gpjax as gpx



class CollapsedBernoulliVariationalGaussian(gpx.variational_families.AbstractVariationalGaussian):
    """
    """

    def __init__(self, posterior, inducing_inputs, jitter=1e-3):
        super().__init__(posterior, inducing_inputs, jitter)
        self.inducing_inputs = jnp.asarray(self.inducing_inputs.value)

    def num_inducing(self):
        """The number of inducing inputs."""
        return self.inducing_inputs.shape[0]

    def predict(self, test_inputs, train_data):
        # Unpack test inputs
        t = test_inputs

        # Unpack training data
        x, y = train_data.X, train_data.y

        # Unpack variational parameters
        z = self.inducing_inputs
        m = self.num_inducing

        # Unpack mean function and kernel
        mean_function = self.posterior.prior.mean_function
        kernel = self.posterior.prior.kernel

        # Mean
        mean_x = mean_function(x)

        Kmn = kernel.cross_covariance(z, x)
        Kmm = add_jitter(kernel.gram(z).to_dense(), 1e-3)
        Kmt = kernel.cross_covariance(z, t)
        Ktt = kernel.gram(t)

        # Compute A
        Sigma_dense = Kmm + jnp.matmul(Kmn, Kmn.T)
        Sigma_dense = add_jitter(Sigma_dense, 1e-3)
        Sigma = psd(Dense(Sigma_dense))
        L1 = lower_cholesky(Sigma)
        M1 = solve(L1, Kmm)
        A = jnp.matmul(M1.T, M1)

        # Compute A_star
        Temp2_dense = Kmm - A
        Temp2_dense = add_jitter(Temp2_dense, 1e-3)
        Temp2 = psd(Dense(Temp2_dense))
        L2 = lower_cholesky(Temp2)
        M2 = solve(L2, Kmm)
        B = jnp.matmul(M2.T, M2)
        Temp3_dense = B
        Temp3_dense = add_jitter(Temp3_dense, 1e-3)
        Temp3 = psd(Dense(Temp3_dense))
        L3 = lower_cholesky(Temp3)
        M3 = solve(L3, Kmt)
        C = jnp.matmul(M3.T, M3)
        A_star = Ktt - C

        # print(jnp.linalg.eigvalsh(Kmm))
        # print(jnp.linalg.eigvalsh(Sigma_dense))
        # print(jnp.linalg.eigvalsh(Temp2_dense))
        # print(jnp.linalg.eigvalsh(Temp3_dense))

        # Compute Mu_star
        front = solve(L1, Kmt)
        back = solve(L1, Kmn)
        almost = jnp.matmul(front.T, back)
        mu_star = jnp.matmul(almost, y - mean_x)

        mean = mu_star
        covariance = A_star

        if hasattr(covariance, "to_dense"):
            covariance = covariance.to_dense()
        covariance = add_jitter(covariance, self.jitter)
        covariance = Dense(covariance)

        return GaussianDistribution(
            loc=jnp.atleast_1d(mean.squeeze()), scale=covariance
        )

