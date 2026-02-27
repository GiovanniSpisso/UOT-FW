"""
Experiment: 1D Truncated Frank-Wolfe
"""

import numpy as np
import time
from FW_1dim_trunc import PW_FW_dim1_trunc, truncated_cost, UOT_cost_upper, build_c_and_mask


# Parameters
n = 100  # problem size
p = 1    # entropy parameter
R = 3     # truncation radius

# Tolerance parameters
delta = 0.01
eps = 0.001
max_iter = 1000


# Generate test data
np.random.seed(0)
mu = np.random.randint(0, 100, size=n).astype(float)
nu = np.random.randint(0, 100, size=n).astype(float)

# Cost function (only in non-truncated entries)
mu = np.ma.masked_equal(mu, 0)
nu = np.ma.masked_equal(nu, 0)
c, mask = build_c_and_mask(n, R, mu, nu)

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
print(f"  c = {c}")
print()

# Run the truncated Frank-Wolfe algorithm
print("Running truncated Frank-Wolfe...")
start_time = time.time()

# Call the function
result = PW_FW_dim1_trunc(mu, nu, M, p, R,
                          max_iter=max_iter, delta=delta, eps=eps)

elapsed_time = time.time() - start_time

print(f"Truncated FW completed in {elapsed_time:.4f} seconds")

xk, grad_xk, x_marg, y_marg, s_i, s_j = result

# Compute and print final cost + truncated cost


cost_trunc = truncated_cost(xk, x_marg, y_marg, c, mu, nu, p, s_i, s_j, R)
print(f"Final UOT cost (truncated): {cost_trunc:.4f}")
cost = UOT_cost_upper(cost_trunc, n, s_i, R, mu)
print(f"Final UOT cost: {cost:.4f}")



print("\nExperiment completed.")
