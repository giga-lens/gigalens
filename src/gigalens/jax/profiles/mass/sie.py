import functools

import jax.numpy as jnp
from jax import jit

import gigalens.profile


class SIE(gigalens.profile.MassProfile):
    _name = "SIE"
    s_scale = 1e-4
    _params = ["theta_E", "e1", "e2", "center_x", "center_y"]

    @functools.partial(jit, static_argnums=(0,))
    def _param_conv(self, theta_E, e1, e2):
        s_scale = 0
        phi = jnp.arctan2(e2, e1) / 2
        c = jnp.minimum(jnp.sqrt(e1 ** 2 + e2 ** 2), 0.9999)
        q = (1 - c) / (1 + c)
        theta_E_conv = theta_E / (jnp.sqrt((1.0 + q ** 2) / (2.0 * q)))
        b = theta_E_conv * jnp.sqrt((1 + q ** 2) / 2)
        s = s_scale * jnp.sqrt((1 + q ** 2) / (2 * q ** 2))
        return b, s, q, phi

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, theta_E, e1, e2, center_x, center_y):
        b, s, q, phi = self._param_conv(theta_E, e1, e2)

        x, y = x - center_x, y - center_y
        x, y = self._rotate(x, y, phi)
        psi = jnp.sqrt(q ** 2 * (s ** 2 + x ** 2) + y ** 2)
        fx = (
                b
                / jnp.sqrt(1.0 - q ** 2)
                * jnp.arctan(jnp.sqrt(1.0 - q ** 2) * x / (psi + s))
        )
        fy = (
                b
                / jnp.sqrt(1.0 - q ** 2)
                * jnp.arctanh(jnp.sqrt(1.0 - q ** 2) * y / (psi + q ** 2 * s))
        )
        fx, fy = self._rotate(fx, fy, -phi)
        return fx, fy

    @functools.partial(jit, static_argnums=(0,))
    def _rotate(self, x, y, phi):
        cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
        return x * cos_phi + y * sin_phi, -x * sin_phi + y * cos_phi
