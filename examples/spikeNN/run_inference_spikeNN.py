from sbi.inference.snpe.snpe_c import SnpeC
import torch
import numpy as np
import pickle
from os.path import join
from pathlib import Path
from examples.spikeNN.spikeNN_utils import SpikeNNPrior, SpikeNNnumpy
from sbi.utils.get_nn_models import posterior_nn


def spikeNNsimulator(w: torch.Tensor) -> torch.Tensor:

    assert (
        w.numel() == 25
    ), "spikeNN needs exactly 25 parameter values and cannot handle batches."

    try:
        ss = (
            SpikeNNnumpy(w.reshape(5, 5)).forward(return_summ_stats=True).reshape(1, 70)
        )
    except RuntimeError:
        ss = np.nan * np.ones((1, 70))

    return torch.as_tensor(ss, dtype=torch.float32)


prior = SpikeNNPrior()
Ws, summary_stats = torch.load(
    join(Path(__file__).parent, "samples_test_100_torch.torch")
).values()
w_o = Ws[0]
x_o = summary_stats[0]

num_simulations = 1000000
num_hidden = 100
num_atoms = 20
num_workers = 10

density_estimator = posterior_nn(
    model="maf",
    prior_mean=prior.mean,
    prior_std=prior.variance,
    x_o_shape=x_o.shape,
    hidden_features=num_hidden,
)

snpe_kwargs = dict(
    simulator=spikeNNsimulator,
    prior=prior,
    x_o=x_o,
    density_estimator=density_estimator,
    z_score_x=True,
    retrain_from_scratch_each_round=False,
    discard_prior_samples=False,
    simulation_batch_size=1,
    num_workers=num_workers,
    num_atoms=num_atoms,
)

infer = SnpeC(**snpe_kwargs)

posterior_estimator = infer(num_rounds=1, num_simulations_per_round=num_simulations)

# Amortized inference for all 100 xs.
samples = []
for ss in summary_stats:
    s = posterior_estimator.sample(101, x=ss.reshape(1, 70))
    samples.append(s)
samples = torch.stack(samples)

# Save the posterior
path = join(Path(__file__).parent, f"posterior_{num_simulations}.torch")
with open(path, "wb") as fh:
    pickle.dump(
        dict(samples=samples, Ws=Ws, sxs=summary_stats), fh
    )
