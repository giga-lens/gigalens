import functools

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax import jit
from lenstronomy.LightModel.Profiles.shapelets import Shapelets as LenstronomyShapelets

import gigalens.profile


class Shapelets(gigalens.profile.LightProfile):
    _name = "SHAPELETS"
    _params = ["beta", "center_x", "center_y"]

    def __init__(self, n_max, use_lstsq=False, interpolate=True):
        super(Shapelets, self).__init__(use_lstsq=use_lstsq)
        del self._params[-1]  # Deletes the amp parameter, to be added again later below with numbering convention
        self.n_layers = int((n_max + 1) * (n_max + 2) / 2)
        self.n_max = n_max
        self.interpolate = interpolate
        n1 = 0
        n2 = 0
        herm_X = []
        herm_Y = []
        self.N1 = []
        self.N2 = []
        decimal_places = len(str(self.n_layers))
        self._amp_names = []
        for i in range(self.n_layers):
            self._params.append(f"amp{str(i).zfill(decimal_places)}")
            self._amp_names.append(f"amp{str(i).zfill(decimal_places)}")
            self.N1.append(n1)
            self.N2.append(n2)
            herm_X.append(LenstronomyShapelets().phi_n(n1, jnp.linspace(-5, 5, 6000)))
            herm_Y.append(LenstronomyShapelets().phi_n(n2, jnp.linspace(-5, 5, 6000)))
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        N = jnp.arange(0, self.n_max + 1, dtype=jnp.float32)
        self.prefactor = 1. / jnp.sqrt(2 ** N * jnp.sqrt(float(jnp.pi)) * jnp.exp(jax.lax.lgamma(N + 1)))
        self.depth = len(self._amp_names)
        self.herm_X = herm_X
        self.herm_Y = herm_Y

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, center_x, center_y, beta, **amp):
        x = (x - center_x) / beta
        y = (y - center_y) / beta
        if self.interpolate:
            ret = tfp.math.interp_regular_1d_grid(x, -5., 5., self.herm_X, fill_value_below=0., fill_value_above=0.)
            ret = ret * tfp.math.interp_regular_1d_grid(y, -5., 5., self.herm_Y, fill_value_below=0.,
                                                        fill_value_above=0.)
            if self.use_lstsq:
                return ret
            else:
                ret = jnp.einsum('i...j,ij->i...j', ret, jnp.stack([amp[x] for x in self._amp_names], axis=0))
                return jnp.sum(ret, axis=0)
        else:
            z = jnp.stack([x,y], axis=0)

            # Calculate phi_n
            ret = jnp.ones((self.n_max + 1, *z.shape))
            ret = ret.at[0].set(jnp.ones_like(z))
            ret = ret.at[1].set(2 * z)
            for n in range(2, self.n_max + 1):
                ret = ret.at[n].set((2 * (z * ret[n - 1] - (n - 1) * ret[n - 2])))
            ret = jnp.einsum('i,i...->i...', self.prefactor, ret)
            XX, YY = ret[:,0,...], ret[:,1,...]
            fac = jnp.exp(-(x ** 2 + y ** 2) / 2)
            if self.use_lstsq:
                return fac * XX[self.N1, ...] * YY[self.N2, ...]
            else:
                return fac * jnp.einsum('ij,i...j->...j', jnp.stack([amp[x] for x in self._amp_names], axis=0),
                                        XX[self.N1, ...] * YY[self.N2, ...])
