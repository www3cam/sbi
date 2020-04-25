from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal

from sbi.inference.snpe import SnpeC
from sbi.simulators.linear_gaussian import linear_gaussian
from sbi.inference.base import simulate_in_batches
from sbi.utils.torchutils import BoxUniform

# use cpu by default
torch.set_default_tensor_type("torch.FloatTensor")


@pytest.mark.parametrize(
    "num_sims", (pytest.param(0, marks=pytest.mark.xfail), 100, 1000)
)
@pytest.mark.parametrize("batch_size", (1, 100, 1000))
def test_simulate_in_batches(
    num_sims,
    batch_size,
    simulator=linear_gaussian,
    prior=BoxUniform(torch.zeros(5), torch.ones(5)),
):
    """Test combinations of num_sims and simulation_batch_size. """

    theta = prior.sample((num_sims,))
    simulate_in_batches(simulator, theta, batch_size)


def test_inference_with_pilot_samples_many_samples():
    """Test whether num_pilot_sims can be same as num_simulations_per_round."""

    num_dim = 3
    x_o = torch.zeros(num_dim)

    prior = MultivariateNormal(
        loc=torch.zeros(num_dim), covariance_matrix=torch.eye(num_dim)
    )

    infer = SnpeC(
        simulator=linear_gaussian, x_o=x_o, prior=prior, simulation_batch_size=100,
    )

    # Run inference.
    num_rounds, num_simulations_per_round = 2, 100
    infer(num_rounds=num_rounds, num_simulations_per_round=num_simulations_per_round)
