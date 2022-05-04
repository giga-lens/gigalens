import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from lenstronomy.LightModel.Profiles.shapelets import Shapelets as LenstronomyShapelets

import gigalens.profile

tfd = tfp.distributions


class Shapelets(gigalens.profile.LightProfile):
    """A flexible light profile using a Hermite polynomial basis. If `interpolate` is set to True, will precalculate
    the Hermite polynomial and interpolate between precalculated values. Otherwise, the Hermite polynomials will be
    evaluated at every call of Shapelets.function()
    """

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
            herm_X.append(LenstronomyShapelets().phi_n(n1, tf.linspace(-5, 5, 6000)))
            herm_Y.append(LenstronomyShapelets().phi_n(n2, tf.linspace(-5, 5, 6000)))
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        N = tf.range(0, self.n_max + 1, dtype=tf.float32)
        self.prefactor = tf.constant(1. / tf.sqrt(2 ** N * tf.sqrt(float(np.pi)) * tf.exp(tf.math.lgamma(N + 1))))
        self.depth = len(self._amp_names)
        self.herm_X = tf.constant(herm_X, dtype=tf.float32)
        self.herm_Y = tf.constant(herm_Y, dtype=tf.float32)

    @tf.function
    def light(self, x, y, center_x, center_y, beta, **amp):
        if self.interpolate:
            x = (x - center_x) / beta
            y = (y - center_y) / beta
            ret = tfp.math.interp_regular_1d_grid(x, -5., 5., self.herm_X, fill_value_below=0., fill_value_above=0.)
            ret = ret * tfp.math.interp_regular_1d_grid(y, -5., 5., self.herm_Y, fill_value_below=0.,
                                                        fill_value_above=0.)
            if self.use_lstsq:
                return ret
            else:
                ret = tf.einsum('i...j,ij->i...j', ret, tf.convert_to_tensor(tf.nest.flatten(amp)))
                return tf.reduce_sum(ret, axis=0)
        else:
            x = (x - center_x) / beta
            y = (y - center_y) / beta
            XX, YY = self.phi_n(x), self.phi_n(y)
            fac = tf.exp(-(x ** 2 + y ** 2) / 2)
            if self.use_lstsq:
                return fac * tf.gather(XX, self.N1, axis=0) * tf.gather(YY, self.N2, axis=0)
            else:
                return fac * tf.einsum('ij,i...j->...j', tf.convert_to_tensor(tf.nest.flatten(amp)),
                                       tf.gather(XX, self.N1, axis=0) * tf.gather(YY, self.N2, axis=0))

    def phi_n(self, x):
        herm_polys = tf.convert_to_tensor([tf.ones_like(x), 2 * x])
        i0 = tf.constant(2.)
        c = lambda i, m: i < self.n_max + 1
        b = lambda i, m: [i + 1, tf.concat([m, (2 * (x * m[-1] - (i - 1) * m[-2]))[tf.newaxis, ...]], axis=0)]
        herm_polys = tf.while_loop(
            c, b, loop_vars=[i0, herm_polys],
            shape_invariants=[i0.get_shape(), tf.TensorShape([None, *herm_polys.shape[1:]])])[1]
        return tf.einsum('i,i...->i...', self.prefactor, herm_polys)
