from abc import ABC, abstractmethod
from typing import Optional, Any, Union

import numpy as np

import gigalens.model
import gigalens.simulator


class ModellingSequenceInterface(ABC):
    """Defines the three steps in modelling:

    1. Multi-starts gradient descent to find the maximum a posteriori (MAP) estimate. See :cite:t:`marti2003,gyorgy2011`.
    2. Variational inference (VI) using the MAP as a starting point. See :cite:t:`hoffman2013,blei2017`. Note that the implementation of variational inference is stochastic variational inference, so VI and SVI are interchangeable.
    3. Hamiltonian Monte Carlo (HMC) using the inverse of the VI covariance matrix as the mass matrix :math:`M`. See :cite:t:`duan1987a, neal2012a`.

    Args:
        phys_model (:obj:`~gigalens.model.PhysicalModel`): The physical model of the lensing system that we want to fit
        prob_model (:obj:`~gigalens.model.ProbabilisticModel`): The probabilistic model of the data we are fitting
        sim_config (:obj:`~gigalens.simulator.SimulatorConfig`): Parameters for image simulation (e.g., pixel scale)
    """

    def __init__(
            self,
            phys_model: gigalens.model.PhysicalModel,
            prob_model: gigalens.model.ProbabilisticModel,
            sim_config: gigalens.simulator.SimulatorConfig,
    ):
        self.phys_model = phys_model
        self.prob_model = prob_model
        self.sim_config = sim_config

    @abstractmethod
    def MAP(
            self,
            optimizer,
            start: Optional[Any],
            n_samples: int,
            num_steps: int,
            seed: Optional[Any],
    ):
        """Finds maximum a posteriori (MAP) estimates for the parameters. See Section 2.3 in `our paper <https://arxiv.org/abs/2202.07663>`__.

        Args:
            optimizer: An optimizer object with which to run MAP. Adam or variants thereof are recommended, using a
                decaying learning rate
            start: Samples from which to start optimization. If none are provided, optimization will be started by
                sampling directly from the prior
            n_samples (int): Number of samples with which to run multi-starts gradient descent
            num_steps (int): Number of gradient descent steps
            seed: A random seed (only necessary if ``start`` is not specified)

        Returns:
            The *unconstrained* parameters of all ``n_samples`` samples after running ``num_steps`` of optimization.
        """
        pass

    @abstractmethod
    def SVI(self, optimizer, start, n_vi: int, num_steps: int, init_scales: Union[float, np.array], seed: Any):
        """Runs stochastic variational inference (SVI) to characterize the posterior scales. Currently, only
        multi-variate Gaussian ansatz is supported. Note that the implementation of variational inference
        is stochastic variational inference, so VI and SVI are interchangeable. This is roughly equivalent to
        taking the Hessian of the log posterior at the MAP. However, in our experience, the Hessian can become unstable
        in high dimensions (in cases of very small eigenvalues). See Section 2.4 in `our paper <https://arxiv.org/abs/2202.07663>`__.

        Args:
            optimizer: An optimizer with which to minimize the ELBO loss. Adam or variants thereof are recommended,
                using slow learning rate warm-up.
            start: Initial guess for posterior mean. Must be shape `(1,d)`, where `d` is the number of parameters.
                Convention is that it is in unconstrained parameter space.
            n_vi (int): Number of samples with which to approximate the ELBO loss
            num_steps (int): Number of optimization steps
            init_scales (float or :obj:`np.array`): Initial VI standard deviation guess
            seed: A random seed for drawing samples from the posterior ansatz

        Returns:
            The fitter posterior in *unconstrained* space
        """
        pass

    @abstractmethod
    def HMC(
            self,
            q_z,
            init_eps: float,
            init_l: int,
            n_hmc: int,
            num_burnin_steps: int,
            num_results: int,
            max_leapfrog_steps: int,
            seed: Any,
    ):
        """Runs Hamiltonian Monte Carlo (HMC) to draw posterior samples. See Section 2.5 in `our paper <https://arxiv.org/abs/2202.07663>`__.

        Args:
            q_z: Fitted posterior from SVI. Used to calculate the mass matrix :math:`M` for preconditioned HMC.
                Convention is that ``q_z`` is an approximation of the *unconstrained* posterior.
            init_eps (float): Initial step size :math:`\epsilon`
            init_l (int): Initial number of leapfrog steps :math:`L`
            n_hmc (int): Number of HMC chains to run in parallel
            num_burnin_steps (int): Number of burn-in steps
            num_results (int): Number of samples to draw from each chain (after burning in)
            max_leapfrog_steps (int): Maximum number of leapfrog steps if :math:`L` is tuned automatically
            seed: A random seed

        Returns:
            Posterior chains in *unconstrained* space
        """
        pass
