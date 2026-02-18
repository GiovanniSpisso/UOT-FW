import numpy as np

idx = np.arange(3)
di = (idx[:, None] - idx[None, :]) ** 2
dj = (idx[:, None] - idx[None, :]) ** 2
c2d = np.sqrt(di[:, None, :, None] + dj[None, :, None, :])

print(c2d)