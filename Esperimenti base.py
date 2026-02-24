import ot
import time
import numpy as np
from FW_1dim import PW_FW_dim1, UOT_cost
from FW_1dim_trunc import PW_FW_dim1_trunc, truncated_cost

# Set precision to 3 decimal places
np.set_printoptions(precision=3, suppress=True)

np.random.seed(0)
# Number of support point
n = 200  
R = 2
p = 1
# Define two positive and discrete measures
mu = np.random.randint(1, 100, size=n)
nu = np.random.randint(1, 100, size=n)

M = n * (np.sum(mu) + np.sum(nu))
delta = 0.001
eps = 0.001
max_iter = 1000

c = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
c_trunc = np.concatenate([np.full(n - abs(k), abs(k)) for k in range(-R + 1, R)])

start_full = time.time()
xk_1d, grad_1d, x_marg_1d, y_marg_1d = PW_FW_dim1(mu, nu, M, p, c, 
                                                  max_iter = max_iter, delta = delta, eps = eps)
elapsed_full = time.time() - start_full
cost_1d_general = UOT_cost(xk_1d, x_marg_1d, y_marg_1d, c, mu, nu, p)
print("Cost of FW_1dim: ", cost_1d_general)
print("FW_1dim time: ", elapsed_full, "seconds")

start_trunc = time.time()
xk_trunc, (grad_xk_x_trunc, grad_xk_s_trunc), x_marg_trunc, y_marg_trunc, s_i, s_j = PW_FW_dim1_trunc(mu, nu, M, p, c_trunc, R,
                                                                                                      max_iter = max_iter, delta = delta, eps = eps)
elapsed_trunc = time.time() - start_trunc

#print("xk_trunc: ", xk_trunc)
cost_trunc = truncated_cost(xk_trunc, x_marg_trunc, y_marg_trunc, c_trunc, mu, nu, p, s_i, s_j, R)
print("Cost of FW_1dim_trunc: ", cost_trunc)
print("FW_1dim_trunc time: ", elapsed_trunc, "seconds")


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
print("\nPOT cost calculated by me: ", UOT_cost(final_plan, x_marg_final, y_marg_final, c, mu, nu, 1))
print("POT unbalanced KL cost:", result_pot.value)
print("POT time: ", elapsed_pot, "seconds")