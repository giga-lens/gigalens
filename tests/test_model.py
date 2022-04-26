import numpy as np
import pytest
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from gigalens.tf.model import ForwardProbModel
from gigalens.model import PhysicalModel
from gigalens.tf.profiles.mass import epl, shear
from gigalens.tf.profiles.light import sersic
from gigalens.tf.inference import ModellingSequence
from gigalens.simulator import SimulatorConfig


def test_bij(default_prior):
    model = ForwardProbModel(default_prior, np.ones((20, 20)), 1, 1)
    sample = default_prior.sample(5, seed=0)
    assert np.allclose(
        tf.nest.flatten(model.bij.forward(model.bij.inverse(sample))),
        tf.nest.flatten(sample),
    )


def test_prior(default_prior):
    model = ForwardProbModel(default_prior, np.ones((20, 20)), 1, 1)
    sample = default_prior.sample(5, seed=0)
    z = model.bij.inverse(sample)
    det_factor = model.unconstraining_bij.forward_log_det_jacobian(
        model.pack_bij.forward(z)
    )
    assert int(tf.size(det_factor)) == 5


def test_map(default_prior, default_physmodel: PhysicalModel, default_data):
    prob_model = ForwardProbModel(default_prior, *default_data)
    sim_config = SimulatorConfig(delta_pix=0.05, num_pix=20)
    model_seq = ModellingSequence(default_physmodel, prob_model, sim_config)
    opt = tf.keras.optimizers.Adam(0)
    start = prob_model.prior.sample(2)
    ret = model_seq.MAP(opt, start, n_samples=2, num_steps=5, seed=0)
    end = prob_model.bij.forward(ret)
    assert np.allclose(tf.nest.flatten(start), tf.nest.flatten(end))

    opt = tf.keras.optimizers.Adam(1e-3)
    ret = model_seq.MAP(opt, start, n_samples=2, num_steps=5, seed=0)
    end = prob_model.bij.forward(ret)
    assert not np.allclose(tf.nest.flatten(start), tf.nest.flatten(end))


def test_vi(default_prior, default_physmodel: PhysicalModel, default_data):
    prob_model = ForwardProbModel(default_prior, *default_data)
    sim_config = SimulatorConfig(delta_pix=0.05, num_pix=20)
    model_seq = ModellingSequence(default_physmodel, prob_model, sim_config)

    optimizer = tf.keras.optimizers.Adam(0)
    start = prob_model.bij.inverse(default_prior.sample(2))[0]
    q_z, losses = model_seq.SVI(optimizer=optimizer, start=start, n_vi=5, num_steps=5)
    assert np.allclose(q_z.mean(), start)

    optimizer = tf.keras.optimizers.Adam(1e-3)
    q_z, losses = model_seq.SVI(optimizer=optimizer, start=start, n_vi=5, num_steps=5)
    assert not np.allclose(q_z.mean(), start)


def test_hmc(default_prior, default_physmodel: PhysicalModel, default_data):
    prob_model = ForwardProbModel(default_prior, *default_data)
    sim_config = SimulatorConfig(delta_pix=0.05, num_pix=20)
    model_seq = ModellingSequence(default_physmodel, prob_model, sim_config)

    optimizer = tf.keras.optimizers.Adam(0)
    start = prob_model.bij.inverse(default_prior.sample(2))[0]
    q_z, losses = model_seq.SVI(optimizer=optimizer, start=start, n_vi=2, num_steps=2)

    num_results = 5
    samples, sample_stats = model_seq.HMC(q_z, n_hmc=3, init_eps=0.3, init_l=3, max_leapfrog_steps=5,
                                          num_burnin_steps=3, num_results=num_results)
    assert len(samples) == num_results
