import numpy as np
from pathlib import Path
from os.path import join
from torch import as_tensor, float32, stack, save

path = join(Path(__file__).parent, "samples_test_100.npz")
data = np.load(path, allow_pickle=True)["data"]

ws = []
sss = []
for w, ss in zip(data[:, 0], data[:, 1]):
    ws.append(as_tensor(w, dtype=float32))
    sss.append(as_tensor(ss, dtype=float32).reshape(70))

path = join(Path(__file__).parent, "samples_test_100_torch.torch")
save(dict(Ws=stack(ws), summary_stats=stack(sss)), path)
