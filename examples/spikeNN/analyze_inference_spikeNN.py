from torch import load
from pathlib import Path
from os.path import join

from sbi.utils.plot.plot import samples_nd

num_simulations = 1000
path = join(Path(__file__).parent, f"posterior_{num_simulations}.torch")
posterior, snpe_kwargs = load(path).values()

samples = posterior.sample(1000)

samples_nd(samples, points=[])

import pdb

pdb.set_trace()
