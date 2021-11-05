import tensorflow as tf
import tensorflow_probability as tfp

from profiles import MassProfile

tfd = tfp.distributions


class SIE(MassProfile):
    _name = 'SIE'
    s_scale = 1e-4
    _params = ['theta_E', 'e1', 'e2', 'center_x', 'center_y']
    _prior = [tfd.Exponential(1), tfd.Normal(0, 0.1), tfd.Normal(0, 0.1), tfd.Normal(0, 0.05),
              tfd.Normal(0, 0.05)]

    @tf.function
    def _param_conv(self, params):
        theta_E, e1, e2, cx, cy = params[0], params[1], params[2], params[3], params[4]
        s_scale = 0
        phi = tf.atan2(e2, e1) / 2
        c = tf.math.minimum(tf.math.sqrt(e1 ** 2 + e2 ** 2), 0.9999)
        q = (1 - c) / (1 + c)
        theta_E_conv = theta_E / (tf.math.sqrt((1. + q ** 2) / (2. * q)))
        b = theta_E_conv * tf.math.sqrt((1 + q ** 2) / 2)
        s = s_scale * tf.math.sqrt((1 + q ** 2) / (2 * q ** 2))
        return b, s, q, phi

    @tf.function
    def deriv(self, x, y, params):
        cx, cy = params[3], params[4]
        b, s, q, phi = self._param_conv(params)

        x, y = x - cx, y - cy
        x, y = self.rotate(x, y, phi)
        psi = tf.math.sqrt(q ** 2 * (s ** 2 + x ** 2) + y ** 2)
        fx = b / tf.math.sqrt(1. - q ** 2) * tf.math.atan(tf.math.sqrt(1. - q ** 2) * x / (psi + s))
        fy = b / tf.math.sqrt(1. - q ** 2) * tf.math.atanh(tf.math.sqrt(1. - q ** 2) * y / (psi + q ** 2 * s))
        fx, fy = self.rotate(fx, fy, -phi)
        return fx, fy
