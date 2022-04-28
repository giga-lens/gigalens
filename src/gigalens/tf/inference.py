import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import (
    distributions as tfd,
    bijectors as tfb,
    experimental as tfe,
)
from tqdm.auto import trange, tqdm

import gigalens.inference
import gigalens.model
import gigalens.tf.simulator


class ModellingSequence(gigalens.inference.ModellingSequenceInterface):
    def MAP(self, optimizer, start=None, n_samples=500, num_steps=350, seed=0):
        tf.random.set_seed(seed)
        start = self.prob_model.prior.sample(n_samples) if start is None else start
        trial = tf.Variable(self.prob_model.bij.inverse(start))
        lens_sim = gigalens.tf.simulator.LensSimulator(
            self.phys_model, self.sim_config, bs=n_samples
        )
        observed_image_size = tf.constant(
            tf.size(self.prob_model.observed_image, out_type=tf.float32)
        )

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                log_prob, square_err = self.prob_model.log_prob(lens_sim, trial)
                agg_loss = tf.reduce_mean(-log_prob / observed_image_size)
            gradients = tape.gradient(agg_loss, [trial])
            optimizer.apply_gradients(zip(gradients, [trial]))
            return square_err

        with trange(num_steps) as pbar:
            for _ in pbar:
                square_err = train_step()
                pbar.set_description(f"Chi Squared: {(np.nanmin(square_err)):.4f}")
        return trial

    def SVI(self, optimizer, start, n_vi=250, init_scales=1e-3, num_steps=500, seed=2):
        tf.random.set_seed(seed)
        lens_sim = gigalens.tf.simulator.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_vi,
        )
        start = tf.squeeze(start)
        scale = (
            np.ones(len(start)).astype(np.float32) * init_scales
            if np.size(init_scales) == 1
            else init_scales
        )
        q_z = tfd.MultivariateNormalTriL(
            loc=tf.Variable(start),
            scale_tril=tfp.util.TransformedVariable(
                np.diag(scale),
                tfp.bijectors.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=1e-6),
                name="stddev",
            ),
        )

        losses = tfp.vi.fit_surrogate_posterior(
            lambda z: self.prob_model.log_prob(lens_sim, z)[0],
            surrogate_posterior=q_z,
            sample_size=n_vi,
            optimizer=optimizer,
            num_steps=num_steps,
        )

        return q_z, losses

    def HMC(
            self,
            q_z,
            init_eps=0.3,
            init_l=3,
            n_hmc=50,
            num_burnin_steps=250,
            num_results=750,
            max_leapfrog_steps=30,
            seed=3,
    ):
        def tqdm_progress_bar_fn(num_steps):
            return iter(tqdm(range(num_steps), desc="", leave=True))

        tf.random.set_seed(seed)
        lens_sim = gigalens.tf.simulator.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_hmc,
        )
        mc_start = q_z.sample(n_hmc)
        cov_estimate = q_z.covariance()

        momentum_distribution = (
            tfe.distributions.MultivariateNormalPrecisionFactorLinearOperator(
                precision_factor=tf.linalg.LinearOperatorLowerTriangular(
                    tf.linalg.cholesky(cov_estimate),
                ),
                precision=tf.linalg.LinearOperatorFullMatrix(cov_estimate),
            )
        )

        @tf.function
        def run_chain():
            num_adaptation_steps = int(num_burnin_steps * 0.8)
            start = tf.identity(mc_start)

            mc_kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
                target_log_prob_fn=lambda z: self.prob_model.log_prob(lens_sim, z)[0],
                momentum_distribution=momentum_distribution,
                step_size=init_eps,
                num_leapfrog_steps=init_l,
            )

            mc_kernel = tfe.mcmc.GradientBasedTrajectoryLengthAdaptation(
                mc_kernel,
                num_adaptation_steps=num_adaptation_steps,
                max_leapfrog_steps=max_leapfrog_steps,
            )
            mc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=mc_kernel, num_adaptation_steps=num_adaptation_steps
            )

            pbar = tfe.mcmc.ProgressBarReducer(
                num_results + num_burnin_steps - 1, progress_bar_fn=tqdm_progress_bar_fn
            )
            mc_kernel = tfe.mcmc.WithReductions(mc_kernel, pbar)

            return tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=start,
                kernel=mc_kernel,
                seed=seed,
            )

        return run_chain()
