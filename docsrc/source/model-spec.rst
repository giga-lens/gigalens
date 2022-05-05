Model Specification
============================
We cannot begin to do inference on our data without writing out, explicitly, a model for the data. Heuristically,
a model ought to be a 'story for how the observed data was generated'. Our model is comprised of two components:

    a. A parameterized physical model for the lensing system, consisting of a model for the mass profile of the main
       lens, potentially a model for the effects of any nearby interlopers, and models for the light profiles of the lens and the source.

    b. A probabilistic model, that consists of a prior for the physical parameters, as well as
       a likelihood function. Defining a likelihood requires a noise model, which for most purposes
       will consist of modeling the noise on a given pixel as the quadrature sum of background Gaussian
       noise :math:`\sigma_{bkg}` (conventionally written as ``background_rms``) and Poisson shot noise
       with exposure time :math:`t_{exp}` (conventionally written as ``exp_time``).

Keep in mind that for any given physical model, there can be a number of valid choices for the probabilistic model.
Therefore, in our implementation, we are careful to keep these two components of the model distinct
(see Section 2.1 of our `paper <https://arxiv.org/abs/2202.07663>`__). The following are our package's high level descriptions for a physical and probabilistic model.

.. automodule:: gigalens.model
    :members:

The TensorFlow implementation of the two above classes are below.

.. automodule:: gigalens.tf.model
    :members:

Prior Specification
----------------------
Once the physical model for the lens is settled, the next step is to specify the prior. For example,
if our model is comprised of an SIS + Shear for the mass model, 2 Sersic's for the lens light, and 1
Sersic for the source light, the prior might look like the following:

.. code-block:: python
    :linenos:

    import tensorflow as tf
    from tensorflow_probability import distributions as tfd

    lens_prior = tfd.JointDistributionSequential(
       [
           tfd.JointDistributionNamed(
               dict(
                   theta_E=tfd.Normal(1.5, 0.25),
                   center_x=tfd.Normal(0, 0.05),
                   center_y=tfd.Normal(0, 0.05),
               )
           ),
           tfd.JointDistributionNamed(
               dict(gamma1=tfd.Normal(0, 0.05), gamma2=tfd.Normal(0, 0.05))
           ),
       ]
    )
    lens_light_prior = tfd.JointDistributionSequential(
       [
           tfd.JointDistributionNamed(
               dict(
                   R_sersic=tfd.LogNormal(tf.math.log(1.0), 0.15),
                   n_sersic=tfd.Uniform(2, 6),
                   center_x=tfd.Normal(0, 0.05),
                   center_y=tfd.Normal(0, 0.05),
                   Ie=tfd.Normal(150.0, 0.5),
               )
           ),
           tfd.JointDistributionNamed(
               dict(
                   R_sersic=tfd.LogNormal(tf.math.log(1.0), 0.15),
                   n_sersic=tfd.Uniform(2, 6),
                   center_x=tfd.Normal(0, 0.05),
                   center_y=tfd.Normal(0, 0.05),
                   Ie=tfd.Normal(150.0, 0.5),
               )
           )
       ]
    )

    source_light_prior = tfd.JointDistributionSequential(
       [
           tfd.JointDistributionNamed(
               dict(
                   R_sersic=tfd.LogNormal(tf.math.log(0.25), 0.15),
                   n_sersic=tfd.Uniform(0.5, 4),
                   center_x=tfd.Normal(0, 0.25),
                   center_y=tfd.Normal(0, 0.25),
                   Ie=tfd.Normal(150.0, 0.5),
               )
           )
       ]
    )

    prior = tfd.JointDistributionSequential(
       [lens_prior, lens_light_prior, source_light_prior]
    )

Note that this is a different model than the one used in our paper. Although this may appear
complex at first, the length is only due to the fact that the model has 20 parameters, each of which must have a prior distribution specified.
All this says is the prior distribution is the product of independent distributions,
that begins with :math:`\theta_E \sim \mathcal{N}(1.5, 0.25)` and ends with
:math:`I_{e,src} \sim \mathcal{N}(150.0, 0.5)`. This way of defining the prior is very
flexible, and allows for dependent distributions as well. The dependence can be specified
with functions:

.. code-block:: python
    :linenos:

    lens_prior = tfd.JointDistributionNamed(
       dict(
           theta_E=tfd.Normal(1.5, 0.25),
           center_x=lambda theta_E: tfd.Normal(theta_E, 0.05),
           center_y=lambda theta_E, center_x: tfd.Normal(theta_E, center_x),
       )
    )

This corresponds to a prior

.. math::
    \theta_E \sim \mathcal{N}(1.5,0.25) \\
    x \sim \mathcal{N}(\theta_E,0.25) \\
    y \sim \mathcal{N}(\theta_E,x) \\

Obviously, this is a completely silly prior, but demonstrates the flexibility of the
:obj:`tfp.distributions.JointDistribution` interface.

Sampling from the prior will produce samples that have an identical structure to the prior.
Sampling twice from the example prior might give something like:

.. code-block:: python
    :linenos:

    [
        [
            {
                "theta_E": [1.2670871, 1.3154074],
                "center_y": [0.057920452, -0.07995515],
                "center_x": [-0.02513125, -0.034906887],
            },
            {
                "gamma2": [0.026222957, 0.0174865],
                "gamma1": [-0.014953673, 0.025195256]
            },
        ],
        [
            {
                "n_sersic": [4.55016, 2.9208045],
                "center_y": [-0.08046845, -0.06934502],
                "center_x": [-0.0336911, -0.013034515],
                "R_sersic": [1.0177809, 0.9325099],
                "Ie": [150.55762, 150.0797],
            },
            {
                "n_sersic": [4.1814566, 3.0877438],
                "center_y": [0.033258155, 0.055594254],
                "center_x": [0.08249615, 0.043057345],
                "R_sersic": [1.2974737, 1.0547239],
                "Ie": [150.98956, 149.52852],
            },
        ],
        [
            {
                "n_sersic": [1.9998379, 2.1873033],
                "center_y": [-0.16878262, 0.24757618],
                "center_x": [0.46696508, 0.3786787],
                "R_sersic": [0.2810259, 0.27873865],
                "Ie": [150.91617, 150.61823],
            }
        ],
    ]

This is a format that is friendly for the ``gigalens`` code to work with.

Unconstraining and Restructuring Parameters
--------------------------------------------

Although the above format is more human-readable, TensorFlow prefers to work directly with Tensors. More precisely,
it prefers to work with *unconstrained* tensors (to ensure differentiability). In the current formulation,
there are certain parameters that are constrained (such as Sersic index). To handle this,
we make use of bijectors that unconstrain parameters and then restructure them into a
convenient form.

.. collapse:: Tensorflow Implementation

    The built-in :obj:`tfp.bijectors.Restructure` does the restructuring conveniently. However,
    this still outputs a list of tensors, when ideally we would like to concatenate all of them into one single Tensor.
    This can be done with :obj:`tfp.bijectors.Split`, but this will completely flatten the parameters into a length
    ``bs*d`` vector. We reshape using the remaining two transforms (:obj:`tfp.bijectors.Reshape` and
    :obj:`tfp.bijectors.Transpose`). Typically, we will use ``z`` to denote this flattened, unconstrained vector.
    Due to our use of :obj:`tfp.bijectors.Split`, the :func:`gigalens.tf.model.ForwardProbModel.log_prob` method
    and :func:`gigalens.tf.model.BackwardProbModel.log_prob` both expect a rank-2 Tensor ``z`` with shape ``(bs,d)``.
    If you are looking to evaluate the log density for just one example, you must expand the first dimension so it has
    shape `(1,d)`.

    .. automodule:: gigalens.tf.model
        :members:

.. collapse:: JAX Implementation

    Since the JAX integration with TPF is still mostly experimental, it seems :obj:`tfp.substrates.jax.bijectors.Split`
    does not work well with traced JAX arrays (which JAX uses with JIT-compiled functions). Therefore, the parameters
    typically remain in list form, rather than a JAX array (although the entire structure is at least flatten into one
    list). Internally, this is managed by converting all parameters to lists before evaluating prior density.

For all intermediate modeling steps, we will always prefer to work in unconstrained space, only converting back to
physical parameter space when absolutely necessary. The final posterior samples from HMC will of course be converted to
physical parameters by applying the bijector. We note that the use of unconstrained parameters means we must
reparameterize the posterior density: this is easily done by multiplying the prior density by the absolute value
of the determinant of the bijector Jacobian.

