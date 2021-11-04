from typing import List

import tensorflow as tf
import tensorflow_probability as tfp
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Util.kernel_util import subgrid_kernel

from gigalens.profiles import LightProfile, MassProfile

tfd = tfp.distributions
tfb = tfp.bijectors
tfem = tfp.experimental.mcmc
tfed = tfp.experimental.distributions


class Model:
    def __init__(self, lenses: List[MassProfile], lens_light: List[LightProfile],
                 source_light: List[LightProfile], name=None, initialize_prior=True):
        self.lenses = lenses
        self.lensLight = lens_light
        self.sourceLight = source_light
        self.name = name

        if initialize_prior:
            priors = []
            for x in [*lenses, *lens_light, *source_light]:
                priors.extend(x.prior)
            self.prior = tfd.Blockwise(priors)
            self.bij = tfp.bijectors.Blockwise(
                [x.experimental_default_event_space_bijector() for x in self.prior.distributions])

    def set_use_lstsq(self, use_lstsq: bool):
        for ll in self.lensLight: ll.set_use_lstsq(use_lstsq)
        for src in self.sourceLight: src.set_use_lstsq(use_lstsq)
        priors = []
        for x in [*self.lenses, *self.lensLight, *self.sourceLight]:
            priors.extend(x.prior)
        self.prior = tfd.Blockwise(priors)
        self.bij = tfp.bijectors.Blockwise(
            [x.experimental_default_event_space_bijector() for x in self.prior.distributions])

    def param_to_dict(self, param):
        if tf.is_tensor(param):
            param = param.numpy()
        lens, lensLight, srcLight = [], [], []
        param_ptr = 0
        for x in self.lenses:
            cnt = len(x.params)
            lens.append({x.params[i]: param[param_ptr + i] for i in range(len(x.params))})
            param_ptr += cnt
        for x in self.lensLight:
            cnt = len(x.params)
            lensLight.append({x.params[i]: param[param_ptr + i] for i in range(len(x.params))})
            param_ptr += cnt
        for x in self.sourceLight:
            cnt = len(x.params)
            srcLight.append({x.params[i]: param[param_ptr + i] for i in range(len(x.params))})
            param_ptr += cnt
        return lens, lensLight, srcLight

    def __str__(self):
        lens = ", ".join([x.name for x in self.lenses if x.name != 'NONE'])
        lensLight = ", ".join([x.name for x in self.lensLight if x.name != 'NONE'])
        srcLight = ", ".join([x.name for x in self.sourceLight if x.name != 'NONE'])
        return f'Lens model: {lens}; Lens light: {lensLight}; Source light: {srcLight}'


class LensSimulator:
    def __init__(self, model: Model, delta_pix=0.05, num_pix=70, bs=1, kernel=None, supersample=1,
                 transform_pix2angle=None):
        self.supersample = int(supersample)
        self.transform_pix2angle = tf.eye(2) * delta_pix if transform_pix2angle is None else transform_pix2angle
        self.conversion_factor = tf.constant(tf.linalg.det(self.transform_pix2angle), dtype=tf.float32)
        self.transform_pix2angle = tf.constant(self.transform_pix2angle, dtype=tf.float32) / float(self.supersample)
        _, _, img_X, img_Y = self.get_coords(self.supersample, num_pix, self.transform_pix2angle)
        self.img_X = tf.constant(tf.repeat(img_X[tf.newaxis, ...], [bs], axis=0), dtype=tf.float32)
        self.img_Y = tf.constant(tf.repeat(img_Y[tf.newaxis, ...], [bs], axis=0), dtype=tf.float32)

        self.model = model
        self.numPix = tf.constant(num_pix)
        self.bs = tf.constant(bs)
        self.depth = tf.constant(sum(map(lambda l: l.n_layers, self.model.lensLight)) + sum(
            map(lambda l: l.n_layers, self.model.sourceLight)))
        self.kernel = None
        self.flat_kernel = None
        if kernel is not None:
            kernel = subgrid_kernel(kernel, supersample, odd=True)[::-1, ::-1, tf.newaxis, tf.newaxis]
            self.kernel = tf.constant(tf.cast(tf.repeat(kernel, self.depth, axis=2), tf.float32), dtype=tf.float32)
            self.flat_kernel = tf.constant(kernel, dtype=tf.float32)

    @staticmethod
    def get_coords(supersample, num_pix, transform_pix2angle):
        lo = tf.range(0, supersample * num_pix, dtype=tf.float32)
        lo = tf.reduce_min(lo - tf.reduce_mean(lo)).numpy()

        ra_at_xy_0, dec_at_xy_0 = tf.squeeze((transform_pix2angle @ ([[lo], [lo]]))).numpy()
        kwargs_pixel_rot = {'nx': supersample * num_pix, 'ny': supersample * num_pix,  # number of pixels per axis
                            'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
                            'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
                            'transform_pix2angle': transform_pix2angle.numpy()}
        pixel_grid_rot = PixelGrid(**kwargs_pixel_rot)

        img_X, img_Y = tf.cast(pixel_grid_rot._x_grid, tf.float32), tf.cast(pixel_grid_rot._y_grid, tf.float32)
        return ra_at_xy_0, dec_at_xy_0, img_X, img_Y

    @tf.function
    def _prepare_params(self, params):
        params = tf.where(tf.math.is_nan(params), tf.zeros_like(params), params)
        params = tf.reshape(params, (self.bs, -1))
        params = tf.transpose(params, perm=[1, 0])
        params = params[..., tf.newaxis, tf.newaxis]
        return params

    @tf.function
    def _beta(self, params, param_ptr=0):
        beta_X, beta_Y = self.img_X, self.img_Y
        for lens in self.model.lenses:
            cnt = len(lens.params)
            f_xi, f_yi = lens.deriv(self.img_X, self.img_Y, params[param_ptr: param_ptr + cnt])
            beta_X, beta_Y = beta_X - f_xi, beta_Y - f_yi
            param_ptr += cnt
        return beta_X, beta_Y, param_ptr

    @tf.function
    def simulate(self, params, no_deflection=False):
        params = self._prepare_params(params)
        beta_X, beta_Y, param_ptr = self._beta(params)
        if no_deflection:
            beta_X, beta_Y = self.img_X, self.img_Y
        img = tf.zeros_like(self.img_X)
        for lensLight in self.model.lensLight:
            cnt = len(lensLight.params)
            img += lensLight.light(self.img_X, self.img_Y, params[param_ptr: param_ptr + cnt])
            param_ptr += cnt
        for src in self.model.sourceLight:
            cnt = len(src.params)
            img += src.light(beta_X, beta_Y, params[param_ptr: param_ptr + cnt])
            param_ptr += cnt
        ret = img[..., tf.newaxis] if self.kernel is None else tf.nn.conv2d(img[..., tf.newaxis], self.flat_kernel,
                                                                            padding='SAME', strides=1)
        ret = tf.nn.avg_pool2d(ret, ksize=self.supersample, strides=self.supersample,
                               padding='SAME') if self.supersample != 1 else ret
        return tf.squeeze(ret) * self.conversion_factor

    @tf.function
    def lstsq_simulate(self, params, data, err_map, return_stacked=False, return_coeffs=False, no_deflection=False):
        params = self._prepare_params(params)
        beta_X, beta_Y, param_ptr = self._beta(params)
        if no_deflection:
            beta_X, beta_Y = self.img_X, self.img_Y
        img = tf.zeros((0, *self.img_X.shape))

        for lensLight in self.model.lensLight:
            cnt = len(lensLight.params)
            light = lensLight.light(self.img_X, self.img_Y, params[param_ptr: param_ptr + cnt])
            light = tf.reshape(light, (lensLight.n_layers, *self.img_X.shape))
            img = tf.concat((img, light), axis=0)
            param_ptr += cnt

        for src in self.model.sourceLight:
            cnt = len(src.params)
            light = src.light(beta_X, beta_Y, params[param_ptr: param_ptr + cnt])
            light = tf.reshape(light, (src.n_layers, *self.img_X.shape))
            img = tf.concat((img, light), axis=0)
            param_ptr += cnt

        img = tf.transpose(img, (1, 2, 3, 0))
        img = tf.reshape(img, (*self.img_X.shape, self.depth))

        img = tf.nn.depthwise_conv2d(img, self.kernel, padding='SAME',
                                     strides=[1, 1, 1, 1]) if self.kernel is not None else img
        ret = tf.nn.avg_pool2d(img, ksize=self.supersample, strides=self.supersample,
                               padding='SAME') if self.supersample != 1 else img
        ret = tf.where(tf.math.is_nan(ret), tf.zeros_like(ret), ret)

        if return_stacked:
            return ret
        W = (1 / err_map)[..., tf.newaxis]
        Y = tf.reshape(tf.cast(data, tf.float32) * tf.squeeze(W), (1, -1, 1))
        X = tf.reshape((ret * W), (self.bs, -1, self.depth))
        Xt = tf.transpose(X, (0, 2, 1))
        coeffs = (tf.linalg.pinv(Xt @ X, rcond=1e-6) @ Xt @ Y)[..., 0]
        if return_coeffs:
            return coeffs
        ret = tf.reduce_sum(ret * coeffs[:, tf.newaxis, tf.newaxis, :], axis=-1)
        return tf.squeeze(ret)
