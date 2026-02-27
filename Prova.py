import numpy as np
import numpy.ma as ma
import FW_1dim_trunc as fw

x = np.asarray([0])
x = np.ma.masked_equal(x, 0)
mask_nonzero = x > 0
result = np.ones_like(x, dtype=float)
result[mask_nonzero] = x[mask_nonzero] * np.log(x[mask_nonzero]) - x[mask_nonzero] + 1

print(result)