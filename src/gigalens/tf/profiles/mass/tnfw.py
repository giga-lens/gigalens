import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import gigalens.profile

tfd = tfp.distributions


class TNFW(gigalens.profile.MassProfile):
    _name = "TNFW"
    _params = ["Rs", "alpha_Rs", "r_trunc", "center_x", "center_y"]

    @tf.function
    def deriv(self, x, y, Rs, alpha_Rs, r_trunc, center_x, center_y):
        rho0 = alpha_Rs / (4.0 * Rs ** 2 * (1.0 + tf.math.log(0.5)))
        dx, dy = (x - center_x), (y - center_y)
        R = tf.math.sqrt(dx ** 2 + dy ** 2)
        R = tf.maximum(R, 0.001 * Rs)
        x = R / Rs
        tau = r_trunc / Rs

        L = tf.math.log(x / (tau + tf.math.sqrt(tau ** 2 + x ** 2)))
        F = self.F(x)
        gx = (
                (tau ** 2)
                / (tau ** 2 + 1) ** 2
                * (
                        (tau ** 2 + 1 + 2 * (x ** 2 - 1)) * F
                        + tau * np.pi
                        + (tau ** 2 - 1) * tf.math.log(tau)
                        + tf.math.sqrt(tau ** 2 + x ** 2) * (-np.pi + L * (tau ** 2 - 1) / tau)
                )
        )
        a = 4 * rho0 * Rs * gx / x ** 2
        return a * dx, a * dy

    def F(self, x):
        # x is r/Rs
        x_shape = tf.shape(x)
        x = tf.reshape(x, (-1,))
        nfwvals = tf.ones_like(x, dtype=tf.float32)
        inds1 = tf.where(x < 1)
        inds2 = tf.where(x > 1)
        x1, x2 = tf.reshape(tf.gather(x, inds1), (-1,)), tf.reshape(
            tf.gather(x, inds2), (-1,)
        )
        nfwvals = tf.tensor_scatter_nd_update(
            nfwvals,
            inds1,
            1 / tf.math.sqrt(1 - x1 ** 2) * tf.math.atanh(tf.math.sqrt(1 - x1 ** 2)),
        )
        nfwvals = tf.tensor_scatter_nd_update(
            nfwvals,
            inds2,
            1 / tf.math.sqrt(x2 ** 2 - 1) * tf.math.atan(tf.math.sqrt(x2 ** 2 - 1)),
        )
        return tf.reshape(nfwvals, x_shape)
