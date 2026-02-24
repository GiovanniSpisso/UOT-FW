import ot
import time
import numpy as np
from FW_1dim import PW_FW_dim1, UOT_cost
from FW_1dim_trunc import PW_FW_dim1_trunc, truncated_cost, UOT_cost_upper, vector_to_matrix

# Set precision to 3 decimal places
np.set_printoptions(precision=3, suppress=True)

np.random.seed(0)
n = 100  
R = 5
p = 1
# Define two positive and discrete measures
mu = np.random.randint(1, 100, size=n)
nu = np.random.randint(1, 100, size=n)

M = n * (np.sum(mu) + np.sum(nu))
delta = 0.001
eps = 0.001
max_iter = 10000

c = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
c_trunc = np.concatenate([np.full(n - abs(k), abs(k)) for k in range(-R + 1, R)])

# FW 1 DIM
start_full = time.time()
xk_1d, grad_1d, x_marg_1d, y_marg_1d = PW_FW_dim1(mu, nu, M, p, c, 
                                                  max_iter = max_iter, delta = delta, eps = eps)
elapsed_full = time.time() - start_full
cost_1d_general = UOT_cost(xk_1d, x_marg_1d, y_marg_1d, c, mu, nu, p)

# FW TRUNC
start_trunc = time.time()
xk_trunc, (grad_xk_x_trunc, grad_xk_s_trunc), x_marg_trunc, y_marg_trunc, s_i, s_j = PW_FW_dim1_trunc(mu, nu, M, p, c_trunc, R,
                                                                                                      max_iter = max_iter, delta = delta, eps = eps)
elapsed_trunc = time.time() - start_trunc

cost_trunc = truncated_cost(xk_trunc, x_marg_trunc, y_marg_trunc, c_trunc, mu, nu, p, s_i, s_j, R)
UOT_cost_upper_val = UOT_cost_upper(cost_trunc, n, s_i, R, mu)
print("FW_1dim_trunc time: ", elapsed_trunc, "seconds")
print("Cost of FW_1dim_trunc: ", cost_trunc)
print("Upper bound on UOT cost from truncation: ", UOT_cost_upper_val)
print("\nS_i and S_j (tot):\n", np.sum(s_i*mu), "\n", np.sum(s_j*nu))
# POT 
X_a = np.arange(n, dtype=np.float64).reshape(-1, 1)
X_b = X_a.copy()

a = mu.ravel()
b = nu.ravel()

start_pot = time.time()
result_pot = ot.solve_sample(X_a, X_b, a, b, metric='euclidean', unbalanced=1)
elapsed_pot = time.time() - start_pot
final_plan = result_pot.plan
x_marg_final = np.sum(final_plan, axis=1)/mu  # sum over j and nu dimensions
y_marg_final = np.sum(final_plan, axis=0)/nu  # sum over i and mu dimensions


# PRINT RESULTS
print('\n' + '='*60)
print("Cost of FW_1dim: ", cost_1d_general)
print("Cost of FW_1dim_trunc: ", cost_trunc)
print("Upper bound on UOT cost from truncation: ", UOT_cost_upper_val)
print("POT cost calculated by me: ", UOT_cost(final_plan, x_marg_final, y_marg_final, c, mu, nu, p))
print("POT unbalanced KL cost:", result_pot.value)
print('='*60)
print("FW_1dim time: ", elapsed_full, "seconds")
print("FW_1dim_trunc time: ", elapsed_trunc, "seconds")
print("POT time: ", elapsed_pot, "seconds")


# Display transportation plan
print("\nFW 1D transportation plan:\n", xk_1d)
print("\nPOT transportation plan:\n", final_plan)
print("\nTruncated FW transportation plan (matrix):\n", vector_to_matrix(xk_trunc, n, R))
print("\nS_i and S_j (tot):\n", np.sum(s_i*mu), "\n", np.sum(s_j*nu))