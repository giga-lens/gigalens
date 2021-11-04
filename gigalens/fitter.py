from abc import ABC

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm.auto import trange

from gigalens.lens_sim import LensSimulator
from gigalens.model import Model

tfd = tfp.distributions
tfb = tfp.bijectors
tfem = tfp.experimental.mcmc
tfed = tfp.experimental.distributions


class FitterABC(ABC):
    """
    Defines the three steps in modelling:
    (1): MAP
    (2): VI using the MAP as a starting point
    (3): HMC using the inverse of the VI covariance matrix as the mass matrix M

    This is an abstract class that can be implemented by implementing the log_prob method
    """

    model: Model
    lens_sim_kwargs: dict

    def log_prob(self, lens_sim: LensSimulator, x):
        """
        :param lens_sim: instance of LensSimulator with batch size bs
        :param x: Array of *unconstrained* parameter vectors with shape [bs, d], where d is the number of parameters
        :return: A vector of the log posterior density with shape [bs]
        """
        return NotImplemented

    def MAP(self, n_samples=500, lr0=3e-2, num_steps=350, seed=0, lr_reduce_factor=20):
        """
        (1) Find the MAP estimate using multi-starts gradient descent.

        :param n_samples: Number of MAP samples. Samples are initialized by sampling from the model prior
        :param lr0: Initial learning rate
        :param num_steps: Number of gradient descent steps
        :param seed: An optional random seed
        :param lr_reduce_factor: Factor to decay the learning rate by over the course of num_steps iterations
        :return: The MAP estimate in physical parameter space
        """
        tf.random.set_seed(seed)
        schedule = tf.keras.optimizers.schedules.PolynomialDecay(lr0, num_steps, lr0 / lr_reduce_factor)
        optimizer = tf.keras.optimizers.Adamax(schedule)
        start = self.model.prior.sample(n_samples)
        trial = tf.Variable(self.model.bij.inverse(start))
        imSim = LensSimulator(**self.lens_sim_kwargs, bs=n_samples)
        hist = []
        chisq_hist = []

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                log_prob, square_err = self.log_prob(imSim, trial)
                loss = -log_prob
                agg_loss = tf.reduce_mean(loss)
            gradients = tape.gradient(agg_loss, [trial])
            optimizer.apply_gradients(zip(gradients, [trial]))
            return loss, square_err

        with trange(num_steps) as pbar:
            for _ in pbar:
                loss, square_err = train_step()
                hist.append(tf.identity(trial))
                chisq_hist.append(square_err)
                pbar.set_description(f'Chi Squared: {(square_err[np.nanargmin(loss)]):.4f}')
        lps = self.log_prob(imSim, trial)[0]
        return self.model.bij.forward(trial).numpy()[tf.argmax(lps).numpy()], (hist, chisq_hist)

    def SVI(self, start, n_vi=250, num_steps=500, lr_increase=3e-4 / 100, lr_max=1e-3):
        """
        (2) Estimate the posterior mean and covariance matrix with stochastic variational inference (SVI).

        :param start: Initial mean (in physical parameter space). Typically the MAP estimate
        :param n_vi: Number of VI samples to use in approximating the ELBO
        :param num_steps: Number of optimization steps for SVI
        :param lr_increase: Amount by which to increase the learning rate per iteration. Initial learning rate is
                            initially small because VI can be unstable in the beginning of optimization due to different
                            parameter scales.
        :param lr_max: Ceiling for learning rate
        :return: Surrogate fitted posterior in *unconstrained space*
        """
        imSim = LensSimulator(**self.lens_sim_kwargs, bs=n_vi)
        tmp = tf.Variable(start)
        with tf.GradientTape() as tape:
            ginv = self.model.bij.inverse(tmp)
        Jinv = np.diag(tape.jacobian(ginv, tmp))  # Inverse jacobian.
        scale = np.ones(len(start)).astype(np.float32) * 1e-3
        scale = Jinv * scale
        q_z = tfd.MultivariateNormalTriL(
            loc=tf.Variable(self.model.bij.inverse(start)),
            scale_tril=tfp.util.TransformedVariable(
                np.diag(scale),
                tfp.bijectors.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=1e-6),
                name="stddev")
        )

        class WarmupLR(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __call__(self, step):
                return tf.clip_by_value(step * lr_increase, 1e-6, lr_max)

        losses = tfp.vi.fit_surrogate_posterior(
            lambda z: self.log_prob(imSim, z)[0],
            surrogate_posterior=q_z,
            sample_size=n_vi,
            optimizer=tf.optimizers.Adamax(WarmupLR()),
            num_steps=num_steps)

        return q_z, losses

    def HMC(self, q_z: tfd.Distribution, init_eps=0.3, init_l=3, n_hmc=50, num_burnin_steps=250, num_results=750):
        """
        (3) Sample from the posterior with HMC

        :param q_z: Surrogate posterior in *unconstrained* space
        :param init_eps: Initial step size
        :param init_l: Initial number of leapfrog steps
        :param n_hmc: Number of HMC walkers
        :param num_burnin_steps: Number of burn-in steps. The step size and number of leapfrog steps are adjusted for
                                 for the first 80% of the burn-in steps
        :param num_results: Number of HMC iterations
        :return: Tuple of posterior samples in *unconstrained* space and various other sampler statistics
        """
        imSim = LensSimulator(**self.lens_sim_kwargs, bs=n_hmc)
        mc_start = q_z.sample(n_hmc)
        cov_estimate = q_z.covariance()

        momentum_distribution = tfed.MultivariateNormalPrecisionFactorLinearOperator(
            precision_factor=tf.linalg.LinearOperatorLowerTriangular(
                tf.linalg.cholesky(cov_estimate),
            ),
            precision=tf.linalg.LinearOperatorFullMatrix(cov_estimate))

        @tf.function
        def run_chain():
            num_adaptation_steps = int(num_burnin_steps * 0.8)
            start = tf.identity(mc_start)

            mc_kernel = tfem.PreconditionedHamiltonianMonteCarlo(
                target_log_prob_fn=lambda z: self.log_prob(imSim, z)[0],
                momentum_distribution=momentum_distribution,
                step_size=init_eps,
                num_leapfrog_steps=init_l)

            mc_kernel = tfem.GradientBasedTrajectoryLengthAdaptation(mc_kernel,
                                                                     num_adaptation_steps=num_adaptation_steps,
                                                                     max_leapfrog_steps=30)
            mc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(inner_kernel=mc_kernel,
                                                                 num_adaptation_steps=num_adaptation_steps)

            pbar = tfem.ProgressBarReducer(num_results + num_burnin_steps - 1, progress_bar_fn=tqdm_progress_bar_fn)
            mc_kernel = tfem.WithReductions(mc_kernel, pbar)

            return tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=start,
                kernel=mc_kernel, )

        return run_chain()


class Fitter(FitterABC):
    """
    Fitter where the likelihood is defined in eqn. (6a)
    """

    def __init__(self, model: Model, delta_pix=1, num_pix=70, kernel=None, supersample=1,
                 transform_pix2angle=None, observed_image=None, background_rms=0, exp_time=1):
        self.model = model
        self.model.set_use_lstsq(False)
        self.lens_sim_kwargs = {
            'model': model,
            'delta_pix': delta_pix,
            'num_pix': num_pix,
            'kernel': kernel,
            'supersample': supersample,
            'transform_pix2angle': transform_pix2angle
        }
        self.background_rms = tf.constant(tf.cast(background_rms, tf.float32), dtype=tf.float32)
        self.exp_time = tf.constant(tf.cast(exp_time, tf.float32), dtype=tf.float32)
        self.observed_image = tf.constant(tf.cast(observed_image, tf.float32), dtype=tf.float32)
        self.N = tf.constant(tf.size(observed_image, out_type=tf.float32))

    @tf.function
    def log_prob(self, lens_sim, z):
        x = self.model.bij.forward(z)
        pred = lens_sim.simulate(x)
        err_map = tf.math.sqrt(self.background_rms ** 2 + tf.clip_by_value(pred, 0, np.inf) / self.exp_time)
        ll = tfd.Independent(tfd.Normal(pred, err_map), reinterpreted_batch_ndims=2).log_prob(self.observed_image)
        ret = ll + self.model.prior.log_prob(x) + self.model.bij.forward_log_det_jacobian(z)
        square_err = tf.reduce_mean(((pred - self.observed_image) / err_map) ** 2, axis=(-2, -1))
        return ret, square_err


class LstSqFitter(FitterABC):
    """
    Fitter where the likelihood is defined in terms of a constant noise map that is estimated from the data. This allows
    the use of least-squares to fit out linear parameters.
    """

    def __init__(self, model: Model, delta_pix=1, num_pix=70, kernel=None, supersample=1,
                 transform_pix2angle=None, observed_image=None, err_map=None):
        self.model = model
        self.model.set_use_lstsq(True)
        self.lens_sim_kwargs = {
            'model': model,
            'delta_pix': delta_pix,
            'num_pix': num_pix,
            'kernel': kernel,
            'supersample': supersample,
            'transform_pix2angle': transform_pix2angle
        }
        observed_image = tf.cast(observed_image, tf.float32)
        err_map = tf.cast(err_map, tf.float32)
        self.observed_image = tf.constant(observed_image, dtype=tf.float32)
        if err_map is None:
            self.err_map = tf.constant(tf.ones_like(self.observed_image), dtype=tf.float32)  # Equal weights
        else:
            self.err_map = tf.constant(err_map, dtype=tf.float32)
        self.observed_data = tfd.Independent(tfd.Normal(loc=observed_image, scale=self.err_map),
                                             reinterpreted_batch_ndims=2)
        self.N = tf.constant(tf.size(observed_image, out_type=tf.float32))

    @tf.function
    def log_prob(self, lens_sim, z):
        x = self.model.bij.forward(z)
        pred = lens_sim.lstsq_simulate(x, self.observed_image, self.err_map)
        ll = self.observed_data.log_prob(pred)
        ret = ll + self.model.prior.log_prob(x) + self.model.bij.forward_log_det_jacobian(z)
        square_err = tf.reduce_mean(((pred - self.observed_image) / self.err_map) ** 2, axis=(1, 2))
        return ret, square_err
