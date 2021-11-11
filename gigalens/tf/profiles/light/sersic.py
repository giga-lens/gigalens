import tensorflow as tf
import tensorflow_probability as tfp

import gigalens.profile

tfd = tfp.distributions


class Sersic(gigalens.profile.LightProfile):
    _name = "SERSIC"
    _params = ["R_sersic", "n_sersic", "center_x", "center_y"]

    @tf.function
    def light(self, x, y, R_sersic, n_sersic, center_x, center_y, Ie=1):
        Ie = 1 if self.use_lstsq else Ie
        R = self._distance(x, y, center_x, center_y)
        bn = 1.9992 * n_sersic - 0.3271
        return Ie * tf.math.exp(-bn * ((R / R_sersic) ** (1 / n_sersic) - 1.0))

    @tf.function
    def _distance(self, x, y, cx, cy, e1=None, e2=None):
        if e1 is None:
            e1 = tf.zeros_like(cx)
        if e2 is None:
            e2 = tf.zeros_like(cx)
        phi = tf.atan2(e2, e1) / 2
        c = tf.math.sqrt(e1 ** 2 + e2 ** 2)
        q = (1 - c) / (1 + c)
        dx, dy = x - cx, y - cy
        cos_phi, sin_phi = tf.math.cos(phi), tf.math.sin(phi)
        xt1 = (cos_phi * dx + sin_phi * dy) * tf.math.sqrt(q)
        xt2 = (-sin_phi * dx + cos_phi * dy) / tf.math.sqrt(q)
        return tf.sqrt(xt1 ** 2 + xt2 ** 2)


class SersicEllipse(Sersic):
    _name = "SERSIC_ELLIPSE"
    _params = ["R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y"]

    @tf.function
    def light(self, x, y, R_sersic, n_sersic, e1, e2, center_x, center_y, Ie=1):
        Ie = 1 if self.use_lstsq else Ie
        R = self._distance(x, y, center_x, center_y, e1, e2)
        bn = 1.9992 * n_sersic - 0.3271
        return Ie * tf.math.exp(-bn * ((R / R_sersic) ** (1 / n_sersic) - 1.0))


class CoreSersic(Sersic):
    _name = "CORE_SERSIC"
    _params = [
        "R_sersic",
        "n_sersic",
        "Rb",
        "alpha",
        "gamma",
        "e1",
        "e2",
        "center_x",
        "center_y",
    ]

    @tf.function
    def light(
        self,
        x,
        y,
        R_sersic,
        n_sersic,
        Rb,
        alpha,
        gamma,
        e1,
        e2,
        center_x,
        center_y,
        Ie=1,
    ):
        Ie = 1 if self.use_lstsq else Ie
        R = self._distance(x, y, center_x, center_y, e1, e2)
        bn = 1.9992 * n_sersic - 0.3271
        result = (
            Ie
            * (1 + (Rb / R) ** alpha) ** (gamma / alpha)
            * tf.math.exp(
                -bn
                * (
                    (R ** alpha + Rb ** alpha)
                    / R_sersic ** alpha ** 1.0
                    / (alpha * n_sersic)
                )
                - 1.0
            )
        )
        return result
