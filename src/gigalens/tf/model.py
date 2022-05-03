import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd, bijectors as tfb

import gigalens.model
import gigalens.tf.simulator


class ForwardProbModel(gigalens.model.ProbabilisticModel):
    """
    Probabilistic model defined using the simulated image as an estimator for the noise variance map. Linear parameters
    *are not* automatically solved for using least squares.

    Attributes:
        observed_image (:obj:`tf.Tensor` or :obj:`numpy.array`): The observed image.
        background_rms (float): The estimated background Gaussian noise level
        exp_time (float): The exposure time (used for calculating Poisson shot noise)
        pack_bij (:obj:`tfp.bijectors.Bijector`): A bijector that reshapes from a tensor to a structured parameter
            object (i.e., dictionaries of parameter values). Does not change the input parameters whatsoever, it only
            reshapes them.
        unconstraining_bij (:obj:`tfp.bijectors.Bijector`): A bijector that maps from physical parameter space to
            unconstrained parameter space. Outputs an identical structure as the input, only maps values to
            unconstrained space.
        bij (:obj:`tfp.bijectors.Bijector`): The composition of :attr:`~gigalens.tf.model.ForwardProbModel.pack_bij` and
            :attr:`~gigalens.tf.model.ForwardProbModel.unconstraining_bij`. The inverse method of ``bij`` will
            unconstrain parameter values and then flatten the entire structure into one tensor.
    """

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
        """Evaluate the reparameterized log posterior density for a batch of unconstrained lens parameters ``z``.
        Simulates the lenses with parameters ``z`` to evaluate the log likelihood. The log prior is calculated by
        calculating the log prior in physical parameter space and adding the log determinant of the Jacobian.

        Notes:
            The log determinant for the default bijector will output a scalar regardless of the batch size of ``z`` due
            to the flattening behavior of :obj:`tfp.bijectors.Split`. To get around this, we use the fact that
            :attr:`~gigalens.tf.model.ForwardProbModel.pack_bij` has determinant 1 (it simply reshapes/restructures the
            input), and the determinant comes purely from
            :attr:`~gigalens.tf.model.ForwardProbModel.unconstraining_bij`.

        Args:
            simulator (:obj:`~gigalens.tf.simulator.LensSimulator`): A simulator object that has the same batch size
                as ``z``.
            z (:obj:`tf.Tensor`): Parameters in unconstrained space with shape ``(bs, d)``, where ``bs`` is the batch
                size and ``d`` is the number of parameters per lens.

        Returns:
            The reparameterized log posterior density, with shape ``(bs,)``.
        """
        x = self.bij.forward(z)
        im_sim = simulator.simulate(x)
        err_map = tf.math.sqrt(self.background_rms ** 2 + tf.clip_by_value(im_sim, 0, np.inf) / self.exp_time)
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
    """
    Probabilistic model defined using the observed image as an estimator for the noise variance map. Linear parameters
    *are* automatically solved for using least squares.

    Attributes:
        observed_image (:obj:`tf.Tensor` or :obj:`numpy.array`): The observed image.
        background_rms (float): The estimated background Gaussian noise level
        exp_time (float): The exposure time (used for calculating Poisson shot noise)
        pack_bij (:obj:`tfp.bijectors.Bijector`): A bijector that reshapes from a tensor to a structured parameter
            object (i.e., dictionaries of parameter values). Does not change the input parameters whatsoever, it only
            reshapes them.
        unconstraining_bij (:obj:`tfp.bijectors.Bijector`): A bijector that maps from physical parameter space to
            unconstrained parameter space. Outputs an identical structure as the input, only maps values to
            unconstrained space.
        bij (:obj:`tfp.bijectors.Bijector`): The composition of :attr:`~gigalens.tf.model.ForwardProbModel.pack_bij` and
            :attr:`~gigalens.tf.model.ForwardProbModel.unconstraining_bij`. The inverse method of ``bij`` will
            unconstrain parameter values and then flatten the entire structure into one tensor.
    """

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
        """Evaluate the reparameterized log posterior density for a batch of unconstrained lens parameters ``z``.
        Simulates the lenses with parameters ``z`` (using least squares to solve for linear light parameters) to
        evaluate the log likelihood. The log prior is calculated by  calculating the log prior in physical parameter
        space and adding the log determinant of the Jacobian.

        Notes:
            The log determinant for the default bijector will output a scalar regardless of the batch size of ``z`` due
            to the flattening behavior of :obj:`tfp.bijectors.Split`. To get around this, we use the fact that
            :attr:`~gigalens.tf.model.ForwardProbModel.pack_bij` has determinant 1 (it simply reshapes/restructures the
            input), and the determinant comes purely from
            :attr:`~gigalens.tf.model.ForwardProbModel.unconstraining_bij`.

        Args:
            simulator (:obj:`~gigalens.tf.simulator.LensSimulator`): A simulator object that has the same batch size
                as ``z``.
            z (:obj:`tf.Tensor`): Parameters in unconstrained space with shape ``(bs, d)``, where ``bs`` is the batch
                size and ``d`` is the number of parameters per lens.

        Returns:
            The reparameterized log posterior density, with shape ``(bs,)``.
        """
        x = self.bij.forward(z)
        im_sim = simulator.lstsq_simulate(x, self.observed_image, self.err_map)
        log_like = self.observed_dist.log_prob(im_sim)
        log_prior = self.prior.log_prob(x) + self.unconstraining_bij.forward_log_det_jacobian(x)
        return log_like + log_prior, tf.reduce_mean(
            ((im_sim - self.observed_image) / self.err_map) ** 2, axis=(-2, -1)
        )
