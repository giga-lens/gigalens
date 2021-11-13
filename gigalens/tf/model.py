import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd, bijectors as tfb

import gigalens.model
import gigalens.tf.simulator


class ForwardProbModel(gigalens.model.ProbabilisticModel):
    def __init__(
        self,
        prior: tfd.Distribution,
        observed_image=None,
        background_rms=None,
        exp_time=None,
    ):
        super(ForwardProbModel, self).__init__(prior)
        self.observed_image = tf.constant(observed_image, dtype=tf.float32)
        self.background_rms = tf.constant(background_rms, dtype=tf.float32)
        self.exp_time = tf.constant(float(exp_time), dtype=tf.float32)
        example = prior.sample(seed=0)
        size = int(tf.size(tf.nest.flatten(example)))
        self.pack_bij = tfb.Chain(
            [
                tfb.pack_sequence_as(example),
                tfb.Split(size),
                tfb.Reshape(event_shape_out=(-1,), event_shape_in=(size, -1)),
                tfb.Transpose(perm=(1, 0)),
            ]
        )
        self.unconstraining_bij = prior.experimental_default_event_space_bijector()
        self.bij = tfb.Chain([self.unconstraining_bij, self.pack_bij])

    @tf.function
    def log_prob(self, simulator: gigalens.tf.simulator.LensSimulator, z):
        x = self.bij.forward(z)
        im_sim = simulator.simulate(x)
        err_map = tf.math.sqrt(self.background_rms ** 2 + im_sim / self.exp_time)
        log_like = tfd.Independent(
            tfd.Normal(im_sim, err_map), reinterpreted_batch_ndims=2
        ).log_prob(self.observed_image)
        log_prior = self.prior.log_prob(
            x
        ) + self.unconstraining_bij.forward_log_det_jacobian(self.pack_bij.forward(z))
        return log_like + log_prior, tf.reduce_mean(
            ((im_sim - self.observed_image) / err_map) ** 2, axis=(-2, -1)
        )


class BackwardProbModel(gigalens.model.ProbabilisticModel):
    def __init__(
        self, prior: tfd.Distribution, observed_image, background_rms, exp_time
    ):
        super(BackwardProbModel, self).__init__(prior)
        err_map = tf.math.sqrt(
            background_rms ** 2 + tf.clip_by_value(observed_image, 0, np.inf) / exp_time
        )
        self.observed_dist = tfd.Independent(
            tfd.Normal(observed_image, err_map), reinterpreted_batch_ndims=2
        )
        self.observed_image = tf.constant(observed_image, dtype=tf.float32)
        self.err_map = tf.constant(err_map, dtype=tf.float32)
        example = prior.sample(seed=0)
        size = int(tf.size(tf.nest.flatten(example)))
        self.pack_bij = tfb.Chain(
            [
                tfb.pack_sequence_as(example),
                tfb.Split(size),
                tfb.Reshape(event_shape_out=(-1,), event_shape_in=(size, -1)),
                tfb.Transpose(perm=(1, 0)),
            ]
        )
        self.unconstraining_bij = prior.experimental_default_event_space_bijector()
        self.bij = tfb.Chain([self.unconstraining_bij, self.pack_bij])

    @tf.function
    def log_prob(self, simulator: gigalens.tf.simulator.LensSimulator, z):
        x = self.bij.forward(z)
        im_sim = simulator.lstsq_simulate(x, self.observed_image, self.err_map)
        log_like = self.observed_dist.log_prob(im_sim)
        log_prior = self.prior.log_prob(
            x
        ) + self.unconstraining_bij.forward_log_det_jacobian(self.pack_bij.forward(z))
        return log_like + log_prior, tf.reduce_mean(
            ((im_sim - self.observed_image) / self.err_map) ** 2, axis=(-2, -1)
        )
