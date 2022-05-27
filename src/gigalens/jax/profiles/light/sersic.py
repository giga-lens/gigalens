import functools

import jax.numpy as jnp
from jax import jit

import gigalens.profile


class Sersic(gigalens.profile.LightProfile):
    _name = "SERSIC"
    _params = ["R_sersic", "n_sersic", "center_x", "center_y"]

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, R_sersic, n_sersic, center_x, center_y, Ie=None):
        R = self.distance(x, y, center_x, center_y)
        bn = 1.9992 * n_sersic - 0.3271
        ret = Ie * jnp.exp(-bn * ((R / R_sersic) ** (1 / n_sersic) - 1.0))
        return ret[jnp.newaxis, ...] if self.use_lstsq else (Ie * ret)

    @functools.partial(jit, static_argnums=(0,))
    def distance(self, x, y, cx, cy, e1=None, e2=None):
        if e1 is None:
            e1 = jnp.zeros_like(cx)
        if e2 is None:
            e2 = jnp.zeros_like(cx)
        phi = jnp.arctan2(e2, e1) / 2
        c = jnp.sqrt(e1 ** 2 + e2 ** 2)
        q = (1 - c) / (1 + c)
        dx, dy = x - cx, y - cy
        cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
        xt1 = (cos_phi * dx + sin_phi * dy) * jnp.sqrt(q)
        xt2 = (-sin_phi * dx + cos_phi * dy) / jnp.sqrt(q)
        return jnp.sqrt(xt1 ** 2 + xt2 ** 2)


class SersicEllipse(Sersic):
    _name = "SERSIC_ELLIPSE"
    _params = ["R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y"]

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, R_sersic, n_sersic, e1, e2, center_x, center_y, Ie=None):
        R = self.distance(x, y, center_x, center_y, e1, e2)
        bn = 1.9992 * n_sersic - 0.3271
        ret = jnp.exp(-bn * ((R / R_sersic) ** (1 / n_sersic) - 1.0))
        return ret[jnp.newaxis, ...] if self.use_lstsq else (Ie * ret)


class CoreSersic(Sersic):
    _name = "CORE_SERSIC"
    _params = [
        "R_sersic",
        "n_sersic",
        "Rb",
        "alpha",
        "gamma",
        "e1",
        "e2",
        "center_x",
        "center_y",
    ]

    @functools.partial(jit, static_argnums=(0,))
    def light(
            self,
            x,
            y,
            R_sersic,
            n_sersic,
            Rb,
            alpha,
            gamma,
            e1,
            e2,
            center_x,
            center_y,
            Ie=None,
    ):
        R = self.distance(x, y, center_x, center_y, e1, e2)
        bn = 1.9992 * n_sersic - 0.3271
        ret = ((1 + (Rb / R) ** alpha) ** (gamma / alpha) * jnp.exp(-bn * (
                (R ** alpha + Rb ** alpha)
                / R_sersic ** alpha ** 1.0
                / (alpha * n_sersic)
        ) - 1.0))
        return ret[jnp.newaxis, ...] if self.use_lstsq else (Ie * ret)
