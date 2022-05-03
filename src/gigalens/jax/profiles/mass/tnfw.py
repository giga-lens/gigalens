import functools

import jax.numpy as jnp
from jax import jit

import gigalens.profile


class TNFW(gigalens.profile.MassProfile):
    _name = "TNFW"
    _params = ["Rs", "alpha_Rs", "r_trunc", "center_x", "center_y"]

    @functools.partial(jit, static_argnums=(0,))
    def deriv(self, x, y, Rs, alpha_Rs, r_trunc, center_x, center_y):
        rho0 = alpha_Rs / (4.0 * Rs ** 2 * (1.0 + jnp.log(0.5)))
        dx, dy = (x - center_x), (y - center_y)
        R = jnp.sqrt(dx ** 2 + dy ** 2)
        R = jnp.maximum(R, 0.001 * Rs)
        x = R / Rs
        tau = r_trunc / Rs

        L = jnp.log(x / (tau + jnp.sqrt(tau ** 2 + x ** 2)))
        F = self.F(x)
        gx = (
                (tau ** 2)
                / (tau ** 2 + 1) ** 2
                * (
                        (tau ** 2 + 1 + 2 * (x ** 2 - 1)) * F
                        + tau * jnp.pi
                        + (tau ** 2 - 1) * jnp.log(tau)
                        + jnp.sqrt(tau ** 2 + x ** 2) * (-jnp.pi + L * (tau ** 2 - 1) / tau)
                )
        )
        a = 4 * rho0 * Rs * gx / x ** 2
        return a * dx, a * dy

    @functools.partial(jit, static_argnums=(0,))
    def F(self, x):
        # x is r/Rs
        x_shape = jnp.shape(x)
        x = jnp.reshape(x, (-1,))
        nfwvals = jnp.ones_like(x, dtype=jnp.float32)
        inds1 = jnp.where(x < 1)
        inds2 = jnp.where(x > 1)
        x1, x2 = jnp.reshape(x[..., inds1]), jnp.reshape(x[..., inds2])
        nfwvals = nfwvals.at[..., inds1].set(1 / jnp.sqrt(1 - x1 ** 2) * jnp.arctanh(jnp.sqrt(1 - x1 ** 2)))
        nfwvals = nfwvals.at[..., inds2].set(1 / jnp.sqrt(x2 ** 2 - 1) * jnp.arctan(jnp.sqrt(x2 ** 2 - 1)))
        return jnp.reshape(nfwvals, x_shape)
