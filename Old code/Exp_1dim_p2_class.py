"""
Experiment: 1D Frank-Wolfe with p=2 (Class-based TriDiagonal implementation)
"""

import numpy as np
import time
import FW_1dim_p2 as fw_class


# Parameters
n = 5  # problem size
p = 2  # entropy parameter (p=2)

# Tolerance parameters
delta = 0.001
eps = 0.001
max_iter = 10


# Generate test data
np.random.seed(1)
mu = np.random.randint(0, 100, size=n).astype(float)
nu = np.random.randint(0, 100, size=n).astype(float)

# Cost function: absolute distance
c = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))

# Set M (upper bound for generalized simplex)
M = n * (np.sum(mu) + np.sum(nu))

print(f"Problem setup:")
print(f"  n = {n}")
print(f"  p = {p}")
print(f"  M (upper bound) = {M:.2f}")
print(f"  max_iter = {max_iter}")
print(f"  delta (convergence tol) = {delta}")
print(f"  eps (direction tol) = {eps}")
print(f"  mu = {mu}")
print(f"  nu = {nu}")
print()

# Run the Frank-Wolfe algorithm with detailed iteration output
print("Running Frank-Wolfe with p=2 (Class-based TriDiagonal)...")
start_time = time.time()

# Initialize
xk, x_marg, y_marg, mask1, mask2 = fw_class.x_init_p2(mu, nu)
grad_xk = fw_class.UOT_grad_p2(x_marg, y_marg, mask1, mask2, c, n)

# Main iteration loop with detailed output
for k in range(max_iter):
    print(f"\n{'='*80}")
    print(f"ITERATION {k}")
    print(f"{'='*80}")
    
    # Display transportation plan as matrix
    xk_matrix = xk.to_dense()
    print("\nTransportation Plan (as matrix):")
    print(xk_matrix)
    
    # Display gradient as matrix
    grad_matrix = grad_xk.to_dense()
    print("\nGradient (as matrix):")
    np.set_printoptions(precision=4, suppress=True)
    print(grad_matrix)
    np.set_printoptions()  # Reset to default
    
    # Display marginals
    print(f"\nMarginals:")
    print(f"  x_marg: {x_marg}")
    print(f"  y_marg: {y_marg}")
    
    # Search direction
    vk = fw_class.direction_class(xk, grad_xk, M, eps)
    FW_i, FW_j, AFW_i, AFW_j = vk
    
    print(f"\nSearch Direction:")
    print(f"  FW matrix coordinates: ({FW_i}, {FW_j})", end="")
    if FW_i != -1:
        print(f", gradient value: {grad_xk.get(FW_i, FW_j):.6f}")
    else:
        print()
    
    print(f"  AFW matrix coordinates: ({AFW_i}, {AFW_j})", end="")
    if AFW_i != -1:
        print(f", gradient value: {grad_xk.get(AFW_i, AFW_j):.6f}")
    else:
        print()
    
    # Gap calculation
    gap = fw_class.gap_calc_class(xk, grad_xk, vk, M)
    print(f"\nGap: {gap:.6f}")
    
    if (gap <= delta) or (vk == (-1, -1, -1, -1)):
        print(f"\nConverged after {k} iterations")
        break
    
    # Apply step update
    if AFW_i != -1:
        gamma0 = xk.get(AFW_i, AFW_j) - 1e-10
        gammak = min(fw_class.opt_step(x_marg, y_marg, c, mu, nu, vk), gamma0)
        xk.update(AFW_i, AFW_j, -gammak)
        x_marg[AFW_i] -= gammak / mu[AFW_i]
        y_marg[AFW_j] -= gammak / nu[AFW_j]
        if FW_i != -1:
            xk.update(FW_i, FW_j, gammak)
            x_marg[FW_i] += gammak / mu[FW_i]
            y_marg[FW_j] += gammak / nu[FW_j]
    else:
        gamma0 = M - np.sum(x_marg * mu) + xk.get(FW_i, FW_j)
        gammak = min(fw_class.opt_step(x_marg, y_marg, c, mu, nu, vk), gamma0)
        xk.update(FW_i, FW_j, gammak)
        x_marg[FW_i] += gammak / mu[FW_i]
        y_marg[FW_j] += gammak / nu[FW_j]
    
    print(f"\nStepsize (gamma): {gammak:.6f}")
    
    # Gradient update
    grad_xk = fw_class.UOT_grad_update_p2(x_marg, y_marg, grad_xk, mask1, mask2, c, vk)

elapsed_time = time.time() - start_time

print(f"\n{'='*80}")
print(f"Frank-Wolfe completed in {elapsed_time:.4f} seconds")
print(f"{'='*80}\n")

# Compute and print final cost
cost = fw_class.cost_p2(xk.to_dense(), x_marg, y_marg, c, mu, nu)
print(f"Final UOT cost: {cost:.6f}")

# Print additional information
print(f"\nTransportation Plan (as matrix):")
print(xk.to_dense())

print(f"\nMarginals:")
print(f"  x_marg: {x_marg}")
print(f"  y_marg: {y_marg}")

# Verify constraints
total_mass = xk.sum()
print(f"\nConstraint verification:")
print(f"  Total mass (should be <= M={M:.2f}): {total_mass:.6f}")
print(f"  Sum(x_marg * mu): {np.sum(x_marg * mu):.6f}")
print(f"  Sum(y_marg * nu): {np.sum(y_marg * nu):.6f}")

print("\nExperiment completed.")
