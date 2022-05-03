import functools

import jax.numpy as jnp
from jax import jit

import gigalens.profile


class SIS(gigalens.profile.MassProfile):
    _name = "SIS"
    _params = ["theta_E", "center_x", "center_y"]

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, theta_E, center_x, center_y):
        dx, dy = x - center_x, y - center_y
        R = jnp.sqrt(dx ** 2 + dy ** 2)
        a = jnp.where(R == 0, 0.0, theta_E / R)
        return a * dx, a * dy
