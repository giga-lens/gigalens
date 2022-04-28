Lens Simulation
=========================

The backbone of our modeling code is the lens simulation. Our lens simulation interface is very simple.

.. autoclass:: gigalens.simulator.LensSimulatorInterface
    :members:

There are only two methods that require implementation: a ``simulate`` method and a ``lstsq_simulate`` method. The only
difference between the two is that `lstsq_simulate` will automatically solve for linear light parameters (such as
Sersic half-light :math:`I_e`) to minimize the chi-squared between the simulated image and some observed image (given a
fixed error map). Configuration options for :obj:`~gigalens.simulator.LensSimulatorInterface` are specified with the
:obj:`~gigalens.simulator.SimulatorConfig` object.

.. autoclass:: gigalens.simulator.SimulatorConfig
    :members:

Our lens simulation code is designed to be fast, differentiable, and vectorized/parallelized. We make judicious use of
TensorFlow's ``tf.function`` and JAX's ``jit`` compilation. Furthermore, the mass and light profiles that we have
implemented are all carefully designed to all stay 'within' TensorFlow or JAX -- that is, absolutely no use of
external libraries such as ``numpy`` or ``scipy`` in methods that are used during simulation time. Below is a list of
currently supported lens and mass profiles. Contributions are welcome!

Available Profiles
**********************************


All parameterizable profiles implement :obj:`~gigalens.profile.Parameterized`.

.. autoclass:: gigalens.profile.Parameterized
    :members:

Mass Profiles
------------------------------------
All mass profiles must implement :obj:`~gigalens.profile.MassProfile`, which essentially requires the definition of the deflection
angle.

.. autoclass:: gigalens.profile.MassProfile
    :members:

Two well-tested mass classes are the ``EPL`` and ``Shear`` profiles.

.. automodule:: gigalens.tf.profiles.mass.epl
    :members:
.. automodule:: gigalens.tf.profiles.mass.shear
    :members:

Experimental
____________________________________
The below are experimental implementations, and although several (such as the ``SIE``) simpler than the ``EPL``,
they have not undergone extensive tests (as opposed to the ``EPL`` and ``Shear``).

.. automodule:: gigalens.tf.profiles.mass.sis
    :members:

.. automodule:: gigalens.tf.profiles.mass.sie
    :members:

.. automodule:: gigalens.tf.profiles.mass.tnfw
    :members:


Light Profiles
------------------------------------
Any light profile must implement :obj:`~gigalens.profile.LightProfile`, which requires
the light amplitude to be defined.

.. autoclass:: gigalens.profile.LightProfile
    :members:

The Sersic profiles are well-tested implementations.

.. automodule:: gigalens.tf.profiles.light.sersic
    :members:

Shapelets are a more flexible light model using a Hermite polynomial basis. They have recently been introduced and are less extensively tested.

.. automodule:: gigalens.tf.profiles.light.shapelets
    :members: