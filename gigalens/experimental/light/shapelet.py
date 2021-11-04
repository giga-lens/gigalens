import tensorflow as tf
import tensorflow_probability as tfp
from lenstronomy.LightModel.Profiles.shapelets import Shapelets as LenstronomyShapelets

from profiles import LightProfile

tfd = tfp.distributions


class ShapeletSet(LightProfile):
    _name = 'SHAPELET_SET'
    _params = ['center_x', 'center_y', 'beta']
    _prior = [tfd.Normal(0, 0.5), tfd.Normal(0, 0.5), tfd.LogNormal(loc=tf.math.log(0.1), scale=0.25)]

    def __init__(self, n_max, use_lstsq=True):
        LightProfile.__init__(self, use_lstsq=use_lstsq, n_layers=int((n_max + 1) * (n_max + 2) / 2))

        n1 = 0
        n2 = 0
        herm_X = []
        herm_Y = []
        for i in range(self.n_layers):
            herm_X.append(LenstronomyShapelets().phi_n(n1, tf.linspace(-5, 5, 5000)))
            herm_Y.append(LenstronomyShapelets().phi_n(n2, tf.linspace(-5, 5, 5000)))
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        self.herm_X = tf.constant(herm_X, dtype=tf.float32)
        self.herm_Y = tf.constant(herm_Y, dtype=tf.float32)

    @tf.function
    def light(self, x, y, params):
        cx, cy, beta = params[0], params[1], params[2]
        x = (x - cx) / beta
        y = (y - cy) / beta
        ret = tfp.math.interp_regular_1d_grid(x, -5., 5., self.herm_X, fill_value_below=0., fill_value_above=0.)
        ret = ret * tfp.math.interp_regular_1d_grid(y, -5., 5., self.herm_Y, fill_value_below=0., fill_value_above=0.)
        if self.use_lstsq:
            return ret
        else:
            ret = tf.math.multiply(ret, params[3:])
            return tf.reduce_sum(ret, axis=0)
