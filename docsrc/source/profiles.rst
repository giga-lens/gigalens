Available Mass and Light Profiles
==================================================
All parameterizable profiles implement :obj:`~gigalens.profile.Parameterized`.

.. autoclass:: gigalens.profile.Parameterized
    :members:


Mass Profiles
------------------------------------
All mass profiles must implement :obj:`~gigalens.profile.MassProfile`, which essentially requires the definition of the deflection
angle.

.. autoclass:: gigalens.profile.MassProfile
    :members:

.. collapse:: Tensorflow Implementations

    .. automodule:: gigalens.tf.profiles.mass.epl
        :members:
    .. automodule:: gigalens.tf.profiles.mass.shear
        :members:
    .. automodule:: gigalens.tf.profiles.mass.sis
        :members:

    .. automodule:: gigalens.tf.profiles.mass.sie
        :members:

.. collapse:: JAX Implementations

    .. automodule:: gigalens.jax.profiles.mass.epl
        :members:
    .. automodule:: gigalens.jax.profiles.mass.shear
        :members:
    .. automodule:: gigalens.jax.profiles.mass.sis
        :members:
    .. automodule:: gigalens.jax.profiles.mass.sie
        :members:


Experimental
____________________________________
The below are experimental implementations that have not undergone extensive tests.

.. automodule:: gigalens.tf.profiles.mass.tnfw
    :members:


Light Profiles
------------------------------------
Any light profile must implement :obj:`~gigalens.profile.LightProfile`, which requires
the light amplitude to be defined.

.. autoclass:: gigalens.profile.LightProfile
    :members:

.. collapse:: Tensorflow Implementations

    .. automodule:: gigalens.tf.profiles.light.sersic
        :members:

    .. automodule:: gigalens.tf.profiles.light.shapelets
        :members:

.. collapse:: JAX Implementations

    .. automodule:: gigalens.jax.profiles.light.sersic
        :members:

    .. automodule:: gigalens.jax.profiles.light.shapelets
        :members:


