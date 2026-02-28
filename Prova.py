import ot
import time
import numpy as np
from FW_1dim_trunc import PW_FW_dim1_trunc, truncated_cost, UOT_cost_upper

# Set precision to 3 decimal places
np.set_printoptions(precision=3, suppress=True)

n        = 500
max_iter = 11
R        = 3
p        = 1
delta    = 0.001
eps      = 0.001

np.random.seed(0)
mu = np.random.randint(1, 100, size=n)
nu = np.random.randint(1, 100, size=n)
M  = n * (np.sum(mu) + np.sum(nu))

c_trunc = np.concatenate([np.full(n - abs(k), abs(k)) for k in range(-R + 1, R)])

xk, (gx, gs), x_marg, y_marg, s_i, s_j = PW_FW_dim1_trunc(mu, nu, M, p, R, max_iter=max_iter, delta=delta, eps=eps)

cost_trunc = truncated_cost(xk, x_marg, y_marg, c_trunc, mu, nu, p, s_i, s_j, R)
print(f"Final truncated cost: {cost_trunc:.10f}")
print("sum si and sum sj: ", np.sum(s_i*mu), np.sum(s_j*nu))