import tensorflow as tf
import tensorflow_probability as tfp

from profiles import MassProfile

tfd = tfp.distributions


class SPEP(MassProfile):
    _name = 'SPEP'
    _params = ['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y']
    _prior = [tfd.Exponential(1), tfd.Normal(2, 0.2), tfd.Normal(0, 0.1), tfd.Normal(0, 0.1), tfd.Normal(0, 0.05),
              tfd.Normal(0, 0.05)]

    @tf.function
    def deriv(self, x, y, params):
        theta_E, gamma, e1, e2, cx, cy = params[0], params[1], params[2], params[3], params[4], params[5]

        phi = tf.atan2(e2, e1) / 2
        c = tf.math.minimum(tf.math.sqrt(e1 ** 2 + e2 ** 2), 0.9999)
        q = (1 - c) / (1 + c)
        phi_e = theta_E * q
        dx, dy = x - cx, y - cy
        E = phi_e / (tf.math.pow((3 - gamma) / 2., 1. / (1 - gamma)) * tf.math.sqrt(q))
        eta = -gamma + 3.0
        cos_phi, sin_phi = tf.cos(phi), tf.sin(phi)
        xt1 = cos_phi * dx + sin_phi * dy
        xt2 = (-sin_phi * dx + cos_phi * dy)
        xt2diff_q2 = xt2 / (q ** 2)
        p2 = xt1 ** 2 + xt2 * xt2diff_q2
        a = tf.where(p2 > 0, p2, 0.000001)
        fac = (1. / eta * tf.math.pow(a / (E * E), eta / 2 - 1)) * 2
        f_x_prim = fac * xt1
        f_y_prim = fac * xt2diff_q2
        return cos_phi * f_x_prim - sin_phi * f_y_prim, sin_phi * f_x_prim + cos_phi * f_y_prim
