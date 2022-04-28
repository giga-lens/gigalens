import tensorflow as tf
import tensorflow_probability as tfp
from lenstronomy.LightModel.Profiles.shapelets import Shapelets as LenstronomyShapelets

import gigalens.profile

tfd = tfp.distributions


class Shapelets(gigalens.profile.LightProfile):
    """A flexible light profile using a Hermite polynomial basis.
    """

    _name = "SHAPELETS"
    _params = ["beta", "center_x", "center_y"]

    def __init__(self, n_max, use_lstsq=False):
        super(Shapelets, self).__init__(use_lstsq=use_lstsq)
        del self._params[-1]  # Deletes the amp parameter, to be added again later below with numbering convention
        self.n_layers = int((n_max + 1) * (n_max + 2) / 2)
        self.n_max = n_max
        n1 = 0
        n2 = 0
        herm_X = []
        herm_Y = []
        decimal_places = len(str(self.n_layers))
        self._amp_names = []
        for i in range(self.n_layers):
            self._params.append(f"amp{str(i).zfill(decimal_places)}")
            self._amp_names.append(f"amp{str(i).zfill(decimal_places)}")
            herm_X.append(LenstronomyShapelets().phi_n(n1, tf.linspace(-5, 5, 6000)))
            herm_Y.append(LenstronomyShapelets().phi_n(n2, tf.linspace(-5, 5, 6000)))
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        self.herm_X = tf.constant(herm_X, dtype=tf.float32)
        self.herm_Y = tf.constant(herm_Y, dtype=tf.float32)

    @tf.function
    def light(self, x, y, center_x, center_y, beta, **amp):
        x = (x - center_x) / beta
        y = (y - center_y) / beta
        ret = tfp.math.interp_regular_1d_grid(x, -5., 5., self.herm_X, fill_value_below=0., fill_value_above=0.)
        ret = ret * tfp.math.interp_regular_1d_grid(y, -5., 5., self.herm_Y, fill_value_below=0., fill_value_above=0.)
        if self.use_lstsq:
            return ret
        else:
            ret = tf.einsum('i...j,ji->i...j', ret, tf.convert_to_tensor(tf.nest.flatten(amp)))
            return tf.reduce_sum(ret, axis=0)
