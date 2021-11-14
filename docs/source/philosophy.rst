Philosophy
====================================



TensorFlow and JAX
------------------------------------
This package provides two 'substrates' for modeling: modeling with TensorFlow and modeling with JAX.
The benefits of modeling with TensorFlow include an easier setup (TensorFlow is fairly standard anywhere
with access to GPUs), and a more mature implementation of algorithms such as variational
inference and HMC in TensorFlow Probability (TFP). However, TensorFlow does not provide an easy
way to parallelize across multiple GPUs. JAX, on the other hand, natively supports multiple
devices, and has experimental support in TFP. For beginners, we recommend starting with the
TensorFlow implementation in ``gigalens.tf`` -- however, for experienced users who may be willing experiment,
``gigalens.jax`` can be significantly faster than ``gigalens.tf``.

Although the two implementations are fundamentally identical, there are some small differences.
The most significant is that the unconstraining bijector used in the JAX implementation is not
able to fully flatten unconstrained parameters into a JAX array. This seems to be due to limited
support for :obj:`tfp.substrates.jax.bijectors.Split`, and this may be resolved in upcoming releases
as the JAX substrate in TFP matures. Similarly, the VI for JAX is implemented
within the ``gigalens`` package, whereas in the TensorFlow implementation, TFP already implements
VI, so we use their built-in method (see :func:`tfp.vi.fit_surrogate_posterior`)