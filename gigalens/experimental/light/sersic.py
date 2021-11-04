import tensorflow as tf
import tensorflow_probability as tfp

from profiles import LightProfile

tfd = tfp.distributions


class Sersic(LightProfile):
    _name = 'SERSIC'
    _params = ['R_sersic', 'n_sersic', 'center_x', 'center_y']
    _prior = [tfd.Exponential(1 / 0.1), tfd.Uniform(0, 10), tfd.Normal(0, 0.3), tfd.Normal(0, 0.3)]

    @tf.function
    def light(self, x, y, params):
        R_sersic, n_sersic, cx, cy = params[0], params[1], params[2], params[3]
        Ie = 1 if self.use_lstsq else params[-1]
        R = self.distance(x, y, cx, cy)
        bn = 1.9992 * n_sersic - 0.3271
        return Ie * tf.math.exp(-bn * ((R / R_sersic) ** (1 / n_sersic) - 1.))


class CoreSersic(LightProfile):
    _name = 'CORE_SERSIC'
    _params = ['R_sersic', 'n_sersic', 'Rb', 'alpha', 'gamma', 'e1', 'e2', 'center_x', 'center_y']
    _prior = [tfd.LogNormal(loc=tf.math.log(0.2), scale=0.2), tfd.Uniform(0, 10),
              tfd.LogNormal(loc=tf.math.log(0.1), scale=0.2), tfd.Uniform(0, 10), tfd.Uniform(0, 10),
              tfd.Normal(0, 0.3), tfd.Normal(0, 0.3), tfd.Normal(0, 0.3), tfd.Normal(0, 0.3)]

    @tf.function
    def light(self, x, y, params):
        R_sersic, n_sersic, Rb, alpha, gamma = params[0], params[1], params[2], params[3], params[4]
        e1, e2, cx, cy = params[5], params[6], params[7], params[8]
        Ie = 1 if self.use_lstsq else params[-1]
        R = self.distance(x, y, cx, cy, e1, e2)
        bn = 1.9992 * n_sersic - 0.3271
        result = Ie * (1 + (Rb / R) ** alpha) ** (gamma / alpha) * tf.math.exp(
            -bn * ((R ** alpha + Rb ** alpha) / R_sersic ** alpha ** 1. / (alpha * n_sersic)) - 1.)
        return result
