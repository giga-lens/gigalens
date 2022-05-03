from typing import List, Dict

import tensorflow as tf
from lenstronomy.Util.kernel_util import subgrid_kernel

import gigalens.model
import gigalens.simulator


class LensSimulator(gigalens.simulator.LensSimulatorInterface):
    def __init__(
            self,
            phys_model: gigalens.model.PhysicalModel,
            sim_config: gigalens.simulator.SimulatorConfig,
            bs: int,
    ):
        super(LensSimulator, self).__init__(phys_model, sim_config, bs)
        self.supersample = int(sim_config.supersample)
        self.transform_pix2angle = (
            tf.eye(2) * sim_config.delta_pix
            if sim_config.transform_pix2angle is None
            else sim_config.transform_pix2angle
        )
        self.conversion_factor = tf.constant(
            tf.linalg.det(self.transform_pix2angle), dtype=tf.float32
        )
        self.transform_pix2angle = tf.constant(
            self.transform_pix2angle, dtype=tf.float32
        ) / float(self.supersample)
        _, _, img_X, img_Y = self.get_coords(
            self.supersample, sim_config.num_pix, self.transform_pix2angle
        )
        self.img_X = tf.constant(
            tf.repeat(img_X[..., tf.newaxis], [bs], axis=-1), dtype=tf.float32
        )
        self.img_Y = tf.constant(
            tf.repeat(img_Y[..., tf.newaxis], [bs], axis=-1), dtype=tf.float32
        )

        self.numPix = tf.constant(sim_config.num_pix)
        self.bs = tf.constant(bs)
        self.depth = tf.constant(
            sum([x.depth for x in self.phys_model.lens_light]) + sum([x.depth for x in self.phys_model.source_light])
        )
        self.kernel = None
        self.flat_kernel = None
        if sim_config.kernel is not None:
            kernel = subgrid_kernel(
                sim_config.kernel, sim_config.supersample, odd=True
            )[::-1, ::-1, tf.newaxis, tf.newaxis]
            self.kernel = tf.constant(
                tf.cast(tf.repeat(kernel, self.depth, axis=2), tf.float32),
                dtype=tf.float32,
            )
            self.flat_kernel = tf.constant(kernel, dtype=tf.float32)

    @tf.function
    def _beta(self, lens_params: List[Dict]):
        beta_x, beta_y = self.img_X, self.img_Y
        for lens, p in zip(self.phys_model.lenses, lens_params):
            f_xi, f_yi = lens.deriv(self.img_X, self.img_Y, **p)
            beta_x, beta_y = beta_x - f_xi, beta_y - f_yi
        return beta_x, beta_y

    @tf.function
    def simulate(self, params, no_deflection=False):
        lens_params = params[0]
        lens_light_params, source_light_params = [], []
        if len(self.phys_model.lens_light) > 0:
            lens_light_params, source_light_params = params[1], params[2]
        else:
            source_light_params = params[1]
        beta_x, beta_y = self._beta(lens_params)
        if no_deflection:
            beta_x, beta_y = self.img_X, self.img_Y
        img = tf.zeros_like(self.img_X)
        for lightModel, p in zip(self.phys_model.lens_light, lens_light_params):
            img += lightModel.light(self.img_X, self.img_Y, **p)
        for lightModel, p in zip(self.phys_model.source_light, source_light_params):
            img += lightModel.light(beta_x, beta_y, **p)
        img = tf.where(tf.math.is_nan(img), tf.zeros_like(img), img)
        img = tf.transpose(img, (2, 0, 1))  # batch size, height, width
        ret = (
            img[..., tf.newaxis]
            if self.kernel is None
            else tf.nn.conv2d(
                img[..., tf.newaxis], self.flat_kernel, padding="SAME", strides=1
            )
        )
        ret = (
            tf.nn.avg_pool2d(
                ret, ksize=self.supersample, strides=self.supersample, padding="SAME"
            )
            if self.supersample != 1
            else ret
        )
        return tf.squeeze(ret) * self.conversion_factor

    @tf.function
    def lstsq_simulate(
            self,
            params,
            observed_image,
            err_map,
            return_stacked=False,
            return_coeffs=False,
            no_deflection=False,
    ):
        lens_params = params[0]
        lens_light_params, source_light_params = [], []
        if len(self.phys_model.lens_light) > 0:
            lens_light_params, source_light_params = params[1], params[2]
        else:
            source_light_params = params[1]
        beta_x, beta_y = self._beta(lens_params)
        if no_deflection:
            beta_x, beta_y = self.img_X, self.img_Y
        img = tf.zeros((0, *self.img_X.shape))
        for lightModel, p in zip(self.phys_model.lens_light, lens_light_params):
            tmp = lightModel.light(self.img_X, self.img_Y, **p)
            img = tf.concat((img, tmp), axis=0, )
        for lightModel, p in zip(self.phys_model.source_light, source_light_params):
            tmp = lightModel.light(beta_x, beta_y, **p)
            img = tf.concat((img, tmp), axis=0)
        img = tf.where(tf.math.is_nan(img), tf.zeros_like(img), img)
        img = tf.transpose(
            img, (3, 1, 2, 0)
        )  # batch size, height, width, number of light components
        img = tf.reshape(
            img,
            (
                self.bs,
                self.numPix * self.supersample,
                self.numPix * self.supersample,
                self.depth,
            ),
        )

        img = (
            tf.nn.depthwise_conv2d(
                img, self.kernel, padding="SAME", strides=[1, 1, 1, 1]
            )
            if self.kernel is not None
            else img
        )
        ret = (
            tf.nn.avg_pool2d(
                img, ksize=self.supersample, strides=self.supersample, padding="SAME"
            )
            if self.supersample != 1
            else img
        )
        ret = tf.where(tf.math.is_nan(ret), tf.zeros_like(ret), ret)

        if return_stacked:
            return ret
        W = (1 / err_map)[..., tf.newaxis]
        Y = tf.reshape(tf.cast(observed_image, tf.float32) * tf.squeeze(W), (1, -1, 1))
        X = tf.reshape((ret * W), (self.bs, -1, self.depth))
        Xt = tf.transpose(X, (0, 2, 1))
        coeffs = (tf.linalg.pinv(Xt @ X, rcond=1e-6) @ Xt @ Y)[..., 0]
        if return_coeffs:
            return coeffs
        ret = tf.reduce_sum(ret * coeffs[:, tf.newaxis, tf.newaxis, :], axis=-1)
        return tf.squeeze(ret)
