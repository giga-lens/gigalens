import tensorflow as tf
import tensorflow_probability as tfp

import gigalens.profile

tfd = tfp.distributions


class SIE(gigalens.profile.MassProfile):
    _name = "SIE"
    s_scale = 1e-4
    _params = ["theta_E", "e1", "e2", "center_x", "center_y"]

    @tf.function
    def _param_conv(self, theta_E, e1, e2):
        s_scale = 0
        phi = tf.atan2(e2, e1) / 2
        c = tf.math.minimum(tf.math.sqrt(e1 ** 2 + e2 ** 2), 0.9999)
        q = (1 - c) / (1 + c)
        theta_E_conv = theta_E / (tf.math.sqrt((1.0 + q ** 2) / (2.0 * q)))
        b = theta_E_conv * tf.math.sqrt((1 + q ** 2) / 2)
        s = s_scale * tf.math.sqrt((1 + q ** 2) / (2 * q ** 2))
        return b, s, q, phi

    @tf.function
    def deriv(self, x, y, theta_E, e1, e2, center_x, center_y):
        b, s, q, phi = self._param_conv(theta_E, e1, e2)

        x, y = x - center_x, y - center_y
        x, y = self._rotate(x, y, phi)
        psi = tf.math.sqrt(q ** 2 * (s ** 2 + x ** 2) + y ** 2)
        fx = (
                b
                / tf.math.sqrt(1.0 - q ** 2)
                * tf.math.atan(tf.math.sqrt(1.0 - q ** 2) * x / (psi + s))
        )
        fy = (
                b
                / tf.math.sqrt(1.0 - q ** 2)
                * tf.math.atanh(tf.math.sqrt(1.0 - q ** 2) * y / (psi + q ** 2 * s))
        )
        fx, fy = self._rotate(fx, fy, -phi)
        return fx, fy

    @tf.function
    def _rotate(self, x, y, phi):
        cos_phi, sin_phi = tf.cos(phi, name=self.name + "rotate-cos"), tf.sin(
            phi, name=self.name + "rotate-sin"
        )
        return x * cos_phi + y * sin_phi, -x * sin_phi + y * cos_phi
