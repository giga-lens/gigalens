import functools

import jax.numpy as jnp
from jax import jit, lax

import gigalens.profile


class EPL(gigalens.profile.MassProfile):
    _name = "EPL"
    _params = ["theta_E", "gamma", "e1", "e2", "center_x", "center_y"]

    def __init__(self, niter=18):
        super().__init__()
        self.niter = niter

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, theta_E, gamma, e1, e2, center_x, center_y):
        phi = jnp.arctan2(e2, e1) / 2
        c = jnp.clip(jnp.sqrt(e1 ** 2 + e2 ** 2), 0, 1)
        q = (1 - c) / (1 + c)
        theta_E_conv = theta_E / (jnp.sqrt((1.0 + q ** 2) / (2.0 * q)))
        b = theta_E_conv * jnp.sqrt((1 + q ** 2) / 2)
        t = gamma - 1

        x, y = x - center_x, y - center_y
        x, y = self.rotate(x, y, phi)

        R = jnp.clip(jnp.sqrt((q * x) ** 2 + y ** 2), 1e-10, 1e10)
        angle = jnp.arctan2(y, q * x)
        f = (1 - q) / (1 + q)
        Cs, Ss = jnp.cos(angle), jnp.sin(angle)
        Cs2, Ss2 = jnp.cos(2 * angle), jnp.sin(2 * angle)

        def update(n, val):
            prefac = -f * (2 * n - (2 - t)) / (2 * n + (2 - t))
            last_x, last_y, fx, fy = val
            last_x, last_y = prefac * (Cs2 * last_x - Ss2 * last_y), prefac * (
                    Ss2 * last_x + Cs2 * last_y
            )
            fx += last_x
            fy += last_y
            return last_x, last_y, fx, fy

        _, _, fx, fy = lax.fori_loop(1, self.niter, update, (Cs, Ss, Cs, Ss))
        prefac = (2 * b) / (1 + q) * ((b / R) ** (t - 1))
        fx, fy = fx * prefac, fy * prefac
        return self.rotate(fx, fy, -phi)

    @functools.partial(jit, static_argnums=(0,))
    def rotate(self, x, y, phi):
        cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
        return x * cos_phi + y * sin_phi, -x * sin_phi + y * cos_phi
