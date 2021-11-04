import tensorflow as tf
import tensorflow_probability as tfp

from gigalens.profiles.light.base import LightProfile

tfd = tfp.distributions


class SersicEllipse(LightProfile):
    _name = 'SERSIC_ELLIPSE'
    _params = ['R_sersic', 'n_sersic', 'e1', 'e2', 'center_x', 'center_y']
    _prior = [tfd.LogNormal(loc=tf.math.log(0.2), scale=0.2), tfd.Uniform(0, 10), tfd.Normal(0, 0.3),
              tfd.Normal(0, 0.3), tfd.Normal(0, 0.5), tfd.Normal(0, 0.5)]

    @tf.function
    def light(self, x, y, params):
        R_sersic, n_sersic, e1, e2, cx, cy = params[0], params[1], params[2], params[3], params[4], params[5]
        Ie = 1 if self.use_lstsq else params[-1]
        R = self.distance(x, y, cx, cy, e1, e2)
        bn = 1.9992 * n_sersic - 0.3271
        return Ie * tf.math.exp(-bn * ((R / R_sersic) ** (1 / n_sersic) - 1.))
