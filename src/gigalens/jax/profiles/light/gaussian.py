import functools

import jax.numpy as jnp
from jax import jit
from jax.scipy.stats import multivariate_normal

import gigalens.profile


class IsotropicGaussian(gigalens.profile.LightProfile):
    """This is in development. It does not yet work."""
    _name = "IsotropicGaussian"
    _params = ["Ie", "sig", "center_x", "center_y"]

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, Ie, sig, center_x, center_y):
        pos = jnp.array([x, y])
        ret = Ie * multivariate_normal.pdf(pos, mean = jnp.array([center_x, center_y]), cov = jnp.array(jnp.eye(2) * sig))
        return ret[jnp.newaxis, ...] if self.use_lstsq else (Ie * ret)


