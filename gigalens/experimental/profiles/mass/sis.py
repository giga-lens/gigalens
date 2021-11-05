import tensorflow as tf
import tensorflow_probability as tfp

from profiles import MassProfile

tfd = tfp.distributions


class SIS(MassProfile):
    _name = 'SIS'
    _params = ['theta_E', 'center_x', 'center_y']
    _prior = [tfd.Exponential(1), tfd.Normal(0, 0.05), tfd.Normal(0, 0.05)]

    @tf.function
    def deriv(self, x, y, params):
        theta_E, cx, cy = params[0], params[1], params[2]
        dx, dy = x - cx, y - cy
        R = tf.math.sqrt(dx ** 2 + dy ** 2)
        a = tf.where(R == 0, 0.0, theta_E / R)
        return a * dx, a * dy
