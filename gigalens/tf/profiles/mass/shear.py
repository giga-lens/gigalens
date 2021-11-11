import tensorflow as tf

import gigalens.profile


class Shear(gigalens.profile.MassProfile):
    _name = "SHEAR"
    _params = ["gamma1", "gamma2"]

    @tf.function
    def deriv(self, x, y, gamma1, gamma2):
        return gamma1 * x + gamma2 * y, gamma2 * x - gamma1 * y
