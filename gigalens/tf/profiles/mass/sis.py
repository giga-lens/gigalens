import tensorflow as tf
import tensorflow_probability as tfp

import gigalens.profile

tfd = tfp.distributions


class SIS(gigalens.profile.MassProfile):
    _name = "SIS"
    _params = ["theta_E", "center_x", "center_y"]

    @tf.function
    def deriv(self, x, y, theta_E, center_x, center_y):
        dx, dy = x - center_x, y - center_y
        R = tf.math.sqrt(dx ** 2 + dy ** 2)
        a = tf.where(R == 0, 0.0, theta_E / R)
        return a * dx, a * dy
