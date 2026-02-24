import time
import numpy as np
import ot

from FW_2dim_trunc import (
	PW_FW_dim2_trunc,
	UOT_cost_upper_dim2,
	cost_matrix_trunc_dim2,
	truncated_cost_dim2,
)
from FW_2dim import UOT_cost


def trunc_2dim(mu, nu, p, R, max_iter=1000, delta=0.001, eps=0.001):
	c_trunc, _ = cost_matrix_trunc_dim2(R)
	M = n * n * (np.sum(mu) + np.sum(nu))

	start = time.time()
	xk, grad_xk, x_marg, y_marg, s_i, s_j = PW_FW_dim2_trunc(
		mu, nu, M, p, R, max_iter=max_iter, delta=delta, eps=eps
	)
	elapsed = time.time() - start

	cost_trunc = truncated_cost_dim2(xk, x_marg, y_marg, c_trunc, mu, nu, p, s_i, s_j, R)
	cost_upper = UOT_cost_upper_dim2(cost_trunc, mu.shape[0], s_i, R)

	return {
		"cost_trunc": cost_trunc,
		"cost_upper": cost_upper,
		"elapsed": elapsed,
		"xk": xk,
		"grad_xk": grad_xk,
		"x_marg": x_marg,
		"y_marg": y_marg,
		"s_i": s_i,
		"s_j": s_j,
	}


np.random.seed(0)
n = 10

mu = np.random.randint(1, 100, size=(n, n)).astype(float)
nu = np.random.randint(1, 100, size=(n, n)).astype(float)

xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
X_a = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
X_b = X_a.copy()

a = mu.ravel()
b = nu.ravel()

start_pot = time.time()
result_pot = ot.solve_sample(X_a, X_b, a, b, metric='euclidean', unbalanced=1)
#ot.lowrank_sinkhorn(X_a, X_b, a, b, reg=0, rank=None, alpha=1e-10, 
#					rescale_cost=True, init='random', reg_init=0.1, seed_init=49, 
#					gamma_init='rescale', numItermax=2000, stopThr=1e-07, warn=True, log=False)
#ot.sinkhorn_unbalanced(a, b, M, reg, reg_m, method='sinkhorn', reg_type='kl')
elapsed_pot = time.time() - start_pot

res_trunc = trunc_2dim(mu, nu, p=1, R=2, max_iter=1000, delta=0.001, eps=0.001)

final_plan = result_pot.plan

idx = np.arange(n)
di = (idx[:, None] - idx[None, :]) ** 2
dj = (idx[:, None] - idx[None, :]) ** 2
c2d = np.sqrt(di[:, None, :, None] + dj[None, :, None, :])
c2d = c2d.reshape(n*n, n*n)  # Reshape to 2D matrix

# Calculate marginals from the final plan
x_marg_final = np.sum(final_plan, axis=1)  # sum over j and nu dimensions
y_marg_final = np.sum(final_plan, axis=0)  # sum over i and mu dimensions
x_marg_final = x_marg_final.reshape(n, n)/mu
y_marg_final = y_marg_final.reshape(n, n)/nu

#print("POT plan:\n", final_plan)
#print("POT x_marg:\n", x_marg_final)
#print("POT y_marg:\n", y_marg_final)
print("\nPOT cost calculated by me: ", UOT_cost(final_plan, x_marg_final, y_marg_final, c2d, mu, nu, 1))
print("POT unbalanced KL cost:", result_pot.value)
print("POT time (s):", elapsed_pot)

#print("\nTrunc x_marg:\n", res_trunc["x_marg"])
#print("Trunc y_marg:\n", res_trunc["y_marg"])
print("\nTrunc 2D cost (truncated):", res_trunc["cost_trunc"])
print("Trunc 2D cost (upper bound):", res_trunc["cost_upper"])
print("Trunc 2D time (s):", res_trunc["elapsed"])