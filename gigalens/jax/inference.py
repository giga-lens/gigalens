import functools

import jax.random
import optax
import tensorflow_probability.substrates.jax as tfp
import time
from jax import jit, pmap
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import (
    distributions as tfd,
    bijectors as tfb,
    experimental as tfe,
)
from tqdm.auto import trange

import gigalens.inference
import gigalens.jax.simulator as sim
import gigalens.model


class ModellingSequence(gigalens.inference.ModellingSequenceInterface):
    def MAP(
            self,
            optimizer: optax.GradientTransformation,
            start=None,
            n_samples=500,
            num_steps=350,
            seed=0,
    ):
        dev_cnt = jax.device_count()
        n_samples = (n_samples // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_samples // dev_cnt,
        )
        seed = jax.random.PRNGKey(seed)

        start = (
            self.prob_model.prior.sample(n_samples, seed=seed)
            if start is None
            else start
        )
        params = jnp.stack(self.prob_model.bij.inverse(start)).T

        opt_state = optimizer.init(params)

        def loss(z):
            lp, chisq = self.prob_model.log_prob(lens_sim, z)
            return -jnp.mean(lp) / jnp.size(self.prob_model.observed_image), chisq

        loss_and_grad = jax.pmap(jax.value_and_grad(loss, has_aux=True))

        def update(params, opt_state):
            splt_params = jnp.array(jnp.split(params, dev_cnt, axis=0))
            (_, chisq), grads = loss_and_grad(splt_params)
            grads = jnp.concatenate(grads, axis=0)
            chisq = jnp.concatenate(chisq, axis=0)

            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return chisq, new_params, opt_state

        with trange(num_steps) as pbar:
            for _ in pbar:
                loss, params, opt_state = update(params, opt_state)
                pbar.set_description(
                    f"Chi-squared: {float(jnp.nanmin(loss, keepdims=True)):.3f}"
                )
        return params

    def SVI(
            self,
            start,
            optimizer: optax.GradientTransformation,
            n_vi=250,
            init_scales=1e-3,
            num_steps=500,
            seed=0,
    ):
        dev_cnt = jax.device_count()
        seeds = jax.random.split(jax.random.PRNGKey(seed), dev_cnt)
        n_vi = (n_vi // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_vi // dev_cnt,
        )
        scale = (
            jnp.diag(jnp.ones(jnp.size(start))) * init_scales
            if jnp.size(init_scales) == 1
            else init_scales
        )
        cov_bij = tfp.bijectors.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=1e-6)
        qz_params = jnp.concatenate(
            [jnp.squeeze(start), cov_bij.inverse(scale)], axis=0
        )
        replicated_params = jax.tree_map(lambda x: jnp.array([x] * dev_cnt), qz_params)

        n_params = jnp.size(start)

        def elbo(qz_params, seed):
            mean = qz_params[:n_params]
            cov = cov_bij.forward(qz_params[n_params:])
            qz = tfd.MultivariateNormalTriL(loc=mean, scale_tril=cov)
            z = qz.sample(n_vi // dev_cnt, seed=seed)
            lps = qz.log_prob(z)
            return jnp.mean(lps - self.prob_model.log_prob(lens_sim, z)[0])

        elbo_and_grad = jit(jax.value_and_grad(jit(elbo), argnums=(0,)))

        @functools.partial(pmap, axis_name="num_devices")
        def get_update(qz_params, seed):
            val, grad = elbo_and_grad(qz_params, seed)
            return jax.lax.pmean(val, axis_name="num_devices"), jax.lax.pmean(
                grad, axis_name="num_devices"
            )

        opt_state = optimizer.init(replicated_params)
        loss_hist = []
        with trange(num_steps) as pbar:
            for step in pbar:
                loss, (grads,) = get_update(replicated_params, seeds)
                loss = float(jnp.mean(loss))
                seeds = jax.random.split(seeds[0], dev_cnt)
                updates, opt_state = optimizer.update(grads, opt_state)
                replicated_params = optax.apply_updates(replicated_params, updates)
                pbar.set_description(f"ELBO: {loss:.3f}")
                loss_hist.append(loss)
        mean = replicated_params[0, :n_params]
        cov = cov_bij.forward(replicated_params[0, n_params:])
        qz = tfd.MultivariateNormalTriL(loc=mean, scale_tril=cov)
        return qz, loss_hist

    def HMC(
            self,
            q_z,
            init_eps=0.3,
            init_l=3,
            n_hmc=50,
            num_burnin_steps=250,
            num_results=750,
            max_leapfrog_steps=30,
            seed=0,
    ):
        dev_cnt = jax.device_count()
        seeds = jax.random.split(jax.random.PRNGKey(seed), dev_cnt)
        n_hmc = (n_hmc // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_hmc // dev_cnt,
        )
        momentum_distribution = tfd.MultivariateNormalFullCovariance(
            loc=jnp.zeros_like(q_z.mean()),
            covariance_matrix=jnp.linalg.inv(q_z.covariance()),
        )

        @jit
        def log_prob(z):
            return self.prob_model.log_prob(lens_sim, z)[0]

        @pmap
        def run_chain(seed):
            start = q_z.sample(n_hmc // dev_cnt, seed=seed)
            num_adaptation_steps = int(num_burnin_steps * 0.8)
            mc_kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
                target_log_prob_fn=log_prob,
                momentum_distribution=momentum_distribution,
                step_size=init_eps,
                num_leapfrog_steps=init_l,
            )

            mc_kernel = tfe.mcmc.GradientBasedTrajectoryLengthAdaptation(
                mc_kernel,
                num_adaptation_steps=num_adaptation_steps,
                max_leapfrog_steps=max_leapfrog_steps,
            )
            mc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=mc_kernel, num_adaptation_steps=num_adaptation_steps
            )

            return tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=start,
                trace_fn=lambda _, pkr: None,
                seed=seed,
                kernel=mc_kernel,
            )

        start = time.time()
        ret = run_chain(seeds)
        end = time.time()
        print(f"Sampling took {(end - start):.1f}s")
        return ret
