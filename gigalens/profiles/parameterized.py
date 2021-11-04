from abc import ABC

import tensorflow as tf


class Parameterized(ABC):
    _name = None
    _prior = []
    _params = []

    def __init__(self, *args, **kwargs):
        self.name = self._name
        self.prior = list(self._prior)
        self.params = list(self._params)

    def __str__(self):
        return self.name

    @tf.function
    def rotate(self, x, y, phi):
        cos_phi, sin_phi = tf.cos(phi, name=self.name + 'rotate-cos'), tf.sin(phi, name=self.name + 'rotate-sin')
        return x * cos_phi + y * sin_phi, -x * sin_phi + y * cos_phi
