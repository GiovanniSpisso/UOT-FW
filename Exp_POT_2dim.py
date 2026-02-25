import time
import numpy as np
import ot

from FW_2dim_trunc import PW_FW_dim2_trunc, UOT_cost_upper_dim2, cost_matrix_trunc_dim2, \
	truncated_cost_dim2
from FW_2dim import PW_FW_dim2, UOT_cost

# Set precision to 3 decimal places
np.set_printoptions(precision=3, suppress=True)

np.random.seed(0)
n = 30
R = 2
p = 1
# Define two positive and discrete measures
mu = np.random.randint(1, 100, size=(n, n))
nu = np.random.randint(1, 100, size=(n, n))

M = n * (np.sum(mu) + np.sum(nu))
delta = 0.01
eps = 0.001
max_iter = 10000

idx = np.arange(n)
di = (idx[:, None] - idx[None, :]) ** 2
dj = (idx[:, None] - idx[None, :]) ** 2
c2d = np.sqrt(di[:, None, :, None] + dj[None, :, None, :])
c2d_POT = c2d.reshape(n*n, n*n)  # Reshape to 2D matrix
c_trunc, _ = cost_matrix_trunc_dim2(R)

# FW 2 DIM
start_full = time.time()
xk_2d, grad_2d, x_marg_2d, y_marg_2d = PW_FW_dim2(mu, nu, M, p, c2d, 
                                                  max_iter = max_iter, delta = delta, eps = eps)
elapsed_full = time.time() - start_full
cost_2d_general = UOT_cost(xk_2d, x_marg_2d, y_marg_2d, c2d, mu, nu, p)

# FW TRUNC
start_trunc = time.time()
xk_trunc, (grad_xk_x_trunc, grad_xk_s_trunc), x_marg_trunc, y_marg_trunc, s_i, s_j = PW_FW_dim2_trunc(mu, nu, M, p, R,
                                                                                                      max_iter = max_iter, delta = delta, eps = eps)
elapsed_trunc = time.time() - start_trunc

cost_trunc = truncated_cost_dim2(xk_trunc, x_marg_trunc, y_marg_trunc, c_trunc, mu, nu, p, s_i, s_j, R)
UOT_cost_upper_val = UOT_cost_upper_dim2(cost_trunc, n, s_i, R, mu)

# POT 
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
final_plan = result_pot.plan

# Calculate marginals from the final plan
x_marg_final = np.sum(final_plan, axis=1)  # sum over j and nu dimensions
y_marg_final = np.sum(final_plan, axis=0)  # sum over i and mu dimensions
x_marg_final = x_marg_final.reshape(n, n)/mu
y_marg_final = y_marg_final.reshape(n, n)/nu


# PRINT RESULTS
print('\n' + '='*60)
print("Cost of FW_2dim: ", cost_2d_general)
print("Cost of FW_2dim_trunc: ", cost_trunc)
print("Upper bound on UOT cost from truncation: ", UOT_cost_upper_val)
print("POT cost calculated by me: ", UOT_cost(final_plan, x_marg_final, y_marg_final, c2d_POT, mu, nu, 1))
print("POT unbalanced KL cost:", result_pot.value)
print('='*60)
print("FW_2dim time: ", elapsed_full, "seconds")
print("FW_2dim_trunc time: ", elapsed_trunc, "seconds")
print("POT time: ", elapsed_pot, "seconds")
print("\nS_i and S_j (tot):\n", np.sum(s_i*mu), "\n", np.sum(s_j*nu))



# Display transportation plan
#print("\nFW 2D transportation plan:\n", xk_2d)
#print("\nPOT transportation plan:\n", final_plan)
# QUESTO NON ESISTE PER ORA print("\nTruncated FW transportation plan (matrix):\n", vector_to_matrix(xk_trunc, n, R))
