import random as rand

import numpy as np
import pytest
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from gigalens.model import PhysicalModel
from gigalens.tf.profiles.light import sersic
from gigalens.tf.profiles.mass import epl, shear


@pytest.fixture
def random():
    rand.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)


@pytest.fixture
def default_prior():
    lens_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    theta_E=tfd.LogNormal(tf.math.log(1.25), 0.25),
                    gamma=tfd.TruncatedNormal(2, 0.25, 1, 3),
                    e1=tfd.Normal(0, 0.1),
                    e2=tfd.Normal(0, 0.1),
                    center_x=tfd.Normal(0, 0.05),
                    center_y=tfd.Normal(0, 0.05),
                )
            ),
            tfd.JointDistributionNamed(
                dict(gamma1=tfd.Normal(0, 0.05), gamma2=tfd.Normal(0, 0.05))
            ),
        ]
    )
    lens_light_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    R_sersic=tfd.LogNormal(tf.math.log(1.0), 0.15),
                    n_sersic=tfd.Uniform(2, 6),
                    e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
                    e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
                    center_x=tfd.Normal(0, 0.05),
                    center_y=tfd.Normal(0, 0.05),
                    Ie=tfd.LogNormal(tf.math.log(500.0), 0.3),
                )
            )
        ]
    )

    source_light_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    R_sersic=tfd.LogNormal(tf.math.log(0.25), 0.15),
                    n_sersic=tfd.Uniform(0.5, 4),
                    e1=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
                    e2=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
                    center_x=tfd.Normal(0, 0.25),
                    center_y=tfd.Normal(0, 0.25),
                    Ie=tfd.LogNormal(tf.math.log(150.0), 0.5),
                )
            )
        ]
    )

    return tfd.JointDistributionSequential(
        [lens_prior, lens_light_prior, source_light_prior]
    )


@pytest.fixture
def default_physmodel():
    return PhysicalModel(
        [epl.EPL(), shear.Shear()], [sersic.SersicEllipse()], [sersic.SersicEllipse()]
    )


@pytest.fixture
def default_data():
    return np.zeros((20, 20)), 0.1, 100  # Image, background_rms, exposure time
