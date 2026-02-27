import numpy as np
import numpy.ma as ma
import FW_1dim_trunc as fw

# Parameters
n = 5  # problem size
p = 2  # entropy parameter (p=2)
R = 2  # truncation radius

# Tolerance parameters
delta = 0.001
eps = 0.001
max_iter = 10

# Generate test data
np.random.seed(0)
mu = np.array([0, 2, 1, 3, 0])
nu = np.array([1, 0, 1, 0, 1])

mu = np.ma.masked_equal(mu, 0)
nu = np.ma.masked_equal(nu, 0)

# Set M (upper bound for generalized simplex)
M = n * (np.sum(mu) + np.sum(nu))

print("Problem setup:")
print(f"  n = {n}")
print(f"  p = {p}")
print(f"  R (truncation radius) = {R}")
print(f"  M (upper bound) = {M:.2f}")
print(f"  max_iter = {max_iter}")
print(f"  delta (convergence tol) = {delta}")
print(f"  eps (direction tol) = {eps}")
print(f"  mu = {mu}")
print(f"  nu = {nu}")
print()

xk, (grad_xk_x, grad_xk_s), x_marg, y_marg, s_i, s_j = fw.PW_FW_dim1_trunc(mu, nu, M, p, R, 
                                                                           max_iter = 100, delta = 0.01, eps = 0.001)  
c, mask = fw.build_c_and_mask(n, R, mu, nu)
cost_trunc = fw.truncated_cost(xk, x_marg, y_marg, c, mu, nu, p, s_i, s_j, R)
print(f"Final UOT cost (truncated): {cost_trunc:.6f}")