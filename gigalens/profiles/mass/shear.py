import tensorflow as tf
import tensorflow_probability as tfp

from gigalens.profiles.mass.base import MassProfile

tfd = tfp.distributions


class Shear(MassProfile):
    _name = 'SHEAR'
    _params = ['gamma1', 'gamma2']
    _prior = [tfd.Normal(0, 0.15), tfd.Normal(0, 0.15)]

    @tf.function
    def deriv(self, x, y, params):
        gamma1, gamma2 = params[0], params[1]
        return gamma1 * x + gamma2 * y, gamma2 * x - gamma1 * y
