import tensorflow as tf
import tensorflow_probability as tfp

import gigalens.profile

tfd = tfp.distributions


class EPL(gigalens.profile.MassProfile):
    """The elliptical power law mass profile, calculated iteratively as in :cite:p:`tessore2015a`.

    Attributes:
        niter (int): Maximum number of iterations for calculating the deflection.
    """

    _name = "EPL"
    _params = ["theta_E", "gamma", "e1", "e2", "center_x", "center_y"]

    def __init__(self, niter=50):
        super().__init__()
        self.niter = niter

    @tf.function
    def deriv(self, x, y, theta_E, gamma, e1, e2, center_x, center_y):
        phi = tf.atan2(e2, e1) / 2
        c = tf.clip_by_value(tf.math.sqrt(e1 ** 2 + e2 ** 2), 0, 1)
        q = (1 - c) / (1 + c)
        theta_E_conv = theta_E / (tf.math.sqrt((1.0 + q ** 2) / (2.0 * q)))
        b = theta_E_conv * tf.math.sqrt((1 + q ** 2) / 2)
        t = gamma - 1

        x, y = x - center_x, y - center_y
        x, y = self._rotate(x, y, phi)

        R = tf.clip_by_value(tf.math.sqrt((q * x) ** 2 + y ** 2), 1e-10, 1e10)
        angle = tf.math.atan2(y, q * x)
        f = (1 - q) / (1 + q)
        Cs, Ss = tf.math.cos(angle), tf.math.sin(angle)
        Cs2, Ss2 = tf.math.cos(2 * angle), tf.math.sin(2 * angle)

        niter = tf.stop_gradient(tf.math.log(1e-12) / tf.math.log(tf.reduce_max(f)) + 2)

        def body(n, p):
            last_x, last_y, f_x, f_y = p
            prefac = -f * (2 * n - (2 - t)) / (2 * n + (2 - t))
            last_x, last_y = prefac * (Cs2 * last_x - Ss2 * last_y), prefac * (
                    Ss2 * last_x + Cs2 * last_y
            )
            return n + 1, (last_x, last_y, f_x + last_x, f_y + last_y)

        _, _, fx, fy = tf.while_loop(
            lambda i, p: i < niter,
            body,
            (1.0, (Cs, Ss, Cs, Ss)),
            maximum_iterations=self.niter,
            swap_memory=True,
            name="EPL_ITERATE",
        )[1]
        prefac = (2 * b) / (1 + q) * tf.math.pow(b / R, t - 1)
        fx, fy = fx * prefac, fy * prefac
        return self._rotate(fx, fy, -phi)

    @tf.function
    def _rotate(self, x, y, phi):
        cos_phi, sin_phi = tf.cos(phi, name=self.name + "rotate-cos"), tf.sin(
            phi, name=self.name + "rotate-sin"
        )
        return x * cos_phi + y * sin_phi, -x * sin_phi + y * cos_phi
