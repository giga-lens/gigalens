from abc import ABC

import tensorflow as tf
import tensorflow_probability as tfp

from gigalens.profiles.parameterized import Parameterized

tfd = tfp.distributions


class LightProfile(Parameterized, ABC):
    def __init__(self, *args, **kwargs):
        Parameterized.__init__(self, *args, **kwargs)
        self.layered = False
        self.use_lstsq = kwargs['use_lstsq'] if 'use_lstsq' in kwargs else True
        self.n_layers = kwargs['n_layers'] if ('n_layers' in kwargs and self.use_lstsq) else 1
        if not self.use_lstsq:
            if self.n_layers == 1:
                self.params.append('amp')
                self.prior.append(tfd.Exponential(0.1))
            else:
                self.params.extend([f'amp{i}' for i in range(self.n_layers)])
                self.prior.extend([tfd.Normal(0, 1) for _ in range(self.n_layers)])

    def set_use_lstsq(self, use_lstsq: bool):
        if use_lstsq and not self.use_lstsq:  # Turn least squares on
            if self.n_layers == 1:
                self.params.append('amp')
                self.prior.append(tfd.Exponential(0.1))
            else:
                self.params.extend([f'amp{i}' for i in range(self.n_layers)])
                self.prior.extend([tfd.Normal(0, 1) for _ in range(self.n_layers)])

        elif not use_lstsq and self.use_lstsq:  # Turn least squares off
            if self.n_layers == 1:
                self.params.pop()
                self.prior.pop()
            else:
                for _ in range(self.n_layers):
                    self.params.pop()
                    self.prior.pop()

    def light(self, x, y, params):
        return NotImplemented

    @tf.function
    def distance(self, x, y, cx, cy, e1=None, e2=None):
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
