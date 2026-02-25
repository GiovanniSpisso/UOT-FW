"""
Experiment: 1D Truncated Frank-Wolfe
"""

import numpy as np
import time
from FW_2dim_trunc import PW_FW_dim2_trunc, truncated_cost_dim2, UOT_cost_upper_dim2, cost_matrix_trunc_dim2


# Parameters
n = 30  # problem size
p = 1    # entropy parameter
R = 3     # truncation radius

# Tolerance parameters
delta = 0.01
eps = 0.001
max_iter = 1000


# Generate test data
np.random.seed(0)
mu = np.random.randint(1, 1001, size=(n,n)).astype(float)
nu = np.random.randint(1, 1001, size=(n,n)).astype(float)

# Create truncated cost matrix
c_trunc, displacement_map = cost_matrix_trunc_dim2(R)

# Set M (upper bound for generalized simplex)
M = 2 * (np.sum(mu) + np.sum(nu))

print(f"Problem setup:")
print(f"  n = {n}")
print(f"  p = {p}")
print(f"  R (truncation radius) = {R}")
print(f"  M (upper bound) = {M}")
print(f"  max_iter = {max_iter}")
print(f"  delta (convergence tol) = {delta}")
print(f"  eps (direction tol) = {eps}")
print()

# Run the truncated Frank-Wolfe algorithm
print("Running truncated Frank-Wolfe...")
start_time = time.time()

# Call the function
result = PW_FW_dim2_trunc(mu, nu, M, p, R, max_iter=max_iter, delta=delta, eps=eps)

elapsed_time = time.time() - start_time

print(f"Truncated FW completed in {elapsed_time:.4f} seconds")

xk, grad_xk, x_marg, y_marg, s_i, s_j = result

# Compute and print final cost + truncated cost


cost_trunc = truncated_cost_dim2(xk, x_marg, y_marg, c_trunc, mu, nu, p, s_i, s_j, R)
print(f"Final UOT cost (truncated): {cost_trunc:.4f}")
cost = UOT_cost_upper_dim2(cost_trunc, n, s_i, R, mu)
print(f"Final UOT cost: {cost:.4f}")

print(f"% difference: {100 * (cost - cost_trunc) / cost_trunc if cost_trunc != 0 else 0}")

print("\nExperiment completed.")