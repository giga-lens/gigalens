import functools

import numpy as np
from jax import jit
from jax import numpy as jnp
from jax import random
from tensorflow_probability.substrates.jax import distributions as tfd, bijectors as tfb

import gigalens.jax.simulator as sim
import gigalens.model


class ForwardProbModel(gigalens.model.ProbabilisticModel):
    def __init__(
            self,
            prior: tfd.Distribution,
            observed_image=None,
            background_rms=None,
            exp_time=None,
    ):
        super(ForwardProbModel, self).__init__(prior)
        self.observed_image = jnp.array(observed_image)
        self.background_rms = jnp.float32(background_rms)
        self.exp_time = jnp.float32(exp_time)
        example = prior.sample(seed=random.PRNGKey(0))
        self.pack_bij = tfb.pack_sequence_as(example)
        self.bij = tfb.Chain(
            [
                prior.experimental_default_event_space_bijector(),
                self.pack_bij,
            ]
        )

    @functools.partial(jit, static_argnums=(0, 1))
    def log_prob(self, simulator: sim.LensSimulator, z):
        z = list(z.T)
        x = self.bij.forward(z)
        im_sim = simulator.simulate(x)
        err_map = jnp.sqrt(self.background_rms ** 2 + im_sim / self.exp_time)
        log_like = tfd.Independent(
            tfd.Normal(im_sim, err_map), reinterpreted_batch_ndims=2
        ).log_prob(self.observed_image)
        log_prior = self.prior.log_prob(x) + self.bij.forward_log_det_jacobian(z)
        return log_like + log_prior, jnp.mean(
            ((im_sim - self.observed_image) / err_map) ** 2, axis=(-2, -1)
        )


class BackwardProbModel(gigalens.model.ProbabilisticModel):
    def __init__(
            self, prior: tfd.Distribution, observed_image, background_rms, exp_time
    ):
        super(BackwardProbModel, self).__init__(prior)
        err_map = jnp.sqrt(
            background_rms ** 2 + jnp.clip(observed_image, 0, np.inf) / exp_time
        )
        self.observed_dist = tfd.Independent(
            tfd.Normal(observed_image, err_map), reinterpreted_batch_ndims=2
        )
        self.observed_image = jnp.array(observed_image)
        self.err_map = jnp.array(err_map)
        example = prior.sample(seed=random.PRNGKey(0))
        self.pack_bij = tfb.pack_sequence_as(example)
        self.bij = tfb.Chain(
            [
                prior.experimental_default_event_space_bijector(),
                self.pack_bij,
            ]
        )

    @functools.partial(jit, static_argnums=(0, 1))
    def log_prob(self, simulator: sim.LensSimulator, z):
        z = list(z.T)
        x = self.bij.forward(z)
        im_sim = simulator.lstsq_simulate(x, self.observed_image, self.err_map)
        log_like = self.observed_dist.log_prob(im_sim)
        log_prior = self.prior.log_prob(x) + self.bij.forward_log_det_jacobian(z)
        return log_like + log_prior, jnp.mean(
            ((im_sim - self.observed_image) / self.err_map) ** 2, axis=(-2, -1)
        )
