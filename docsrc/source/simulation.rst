Lens Simulation
=========================



The backbone of our modeling code is the lens simulation. Our lens simulation interface is very simple.

.. autoclass:: gigalens.simulator.LensSimulatorInterface
    :members:

There are only two methods that require implementation: a ``simulate`` method and a ``lstsq_simulate`` method. The only
difference between the two is that `lstsq_simulate` will automatically solve for linear light parameters (such as
Sersic half-light :math:`I_e`) to minimize the :math:`\chi^2` between the simulated image and some observed image (given a
fixed error map). Configuration options for :obj:`~gigalens.simulator.LensSimulatorInterface` are specified with the
:obj:`~gigalens.simulator.SimulatorConfig` object.

.. autoclass:: gigalens.simulator.SimulatorConfig
    :members:

Our lens simulation code is designed to be fast, differentiable, and vectorized/parallelized. We make judicious use of
TensorFlow's ``tf.function`` and JAX's ``jit`` compilation. Furthermore, the mass and light profiles that we have
implemented are all carefully designed to all stay 'within' TensorFlow or JAX -- that is, absolutely no use of
external libraries such as ``numpy`` or ``scipy`` in methods that are used during simulation time. Below is a list of
currently supported lens and mass profiles. Contributions are welcome!

.. toctree::
    profiles.rst

