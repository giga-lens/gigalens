import tensorflow as tf
import tensorflow_probability as tfp

from profiles import LightProfile

tfd = tfp.distributions


class Constant(LightProfile):
    _name = 'CONSTANT'
    _params = []
    _prior = []

    @tf.function
    def light(self, x, y, params):
        Ie = 1 if self.use_lstsq else params[-1]
        return tf.ones_like(x) * Ie
