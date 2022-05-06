Overview
====================================



TensorFlow and JAX
------------------------------------
This package provides two 'substrates' for modeling: modeling with TensorFlow and modeling with JAX.
The benefits of modeling with TensorFlow include an easier setup (TensorFlow is fairly standard anywhere
with access to GPUs), and a more mature implementation of algorithms such as variational
inference and HMC in TensorFlow Probability (TFP). However, for non-neural network applications, TensorFlow does not provide an easy
way to parallelize across multiple GPUs. JAX, on the other hand, natively supports multiple
devices, and has experimental support in TFP. For beginners, we recommend starting with the
TensorFlow substrate in ``gigalens.tf`` -- however, for experienced users who may be willing to experiment,
``gigalens.jax`` with multiple GPUs can be significantly faster than ``gigalens.tf``.

Although the two implementations are fundamentally identical, there are some small differences. While the flattening
of the parameter vector is automatically done by TFP for the TF substrate, for now the
flattening for the JAX substrate is done by hand (JAX may add this feature in the future, in which
case, we will update our implementation). Similarly, we implement VI for JAX, whereas in the TensorFlow substrate,
TFP already implements VI, so we use their built-in method (see :func:`tfp.vi.fit_surrogate_posterior`). Also, we
note that using classes with TensorFlow can cause ~15% slowdown compared with less modular code. This is due
to the ``@tf.function`` wrapper tracing the ``self`` object in class methods, but we choose this approach
for the ease of use and maintenance.
