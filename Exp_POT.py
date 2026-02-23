import time
import numpy as np
import ot

from FW_2dim_trunc import (
	PW_FW_dim2_trunc,
	UOT_cost_upper_dim2,
	cost_matrix_trunc_dim2,
	truncated_cost_dim2,
)


def trunc_2dim(mu, nu, p=1, R=3, max_iter=500, delta=0.01, eps=0.001):
	c_trunc, _ = cost_matrix_trunc_dim2(R)
	M = 2 * (np.sum(mu) + np.sum(nu))

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
n = 1000

mu = np.random.randint(1, 1001, size=(n, n)).astype(float)
nu = np.random.randint(1, 1001, size=(n, n)).astype(float)

xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
X_a = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
X_b = X_a.copy()

a = mu.ravel()
b = nu.ravel()

start_pot = time.time()
result_pot = ot.solve_sample(X_a, X_b, a, b, unbalanced=1, lazy=True)
#ot.lowrank_sinkhorn(X_a, X_b, a, b, reg=0, rank=None, alpha=1e-10, 
#					rescale_cost=True, init='random', reg_init=0.1, seed_init=49, 
#					gamma_init='rescale', numItermax=2000, stopThr=1e-07, warn=True, log=False)
#ot.sinkhorn_unbalanced(a, b, M, reg, reg_m, method='sinkhorn', reg_type='kl')
elapsed_pot = time.time() - start_pot

res_trunc = trunc_2dim(mu, nu, p=1, R=3, max_iter=500, delta=0.01, eps=0.001)

print("POT unbalanced KL cost:", result_pot.value)
print("POT time (s):", elapsed_pot)
print("Trunc 2D cost (truncated):", res_trunc["cost_trunc"])
print("Trunc 2D cost (upper bound):", res_trunc["cost_upper"])
print("Trunc 2D time (s):", res_trunc["elapsed"])