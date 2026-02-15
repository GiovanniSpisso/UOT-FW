"""
Experiment: 1D Frank-Wolfe (full matrix implementation)
"""

import numpy as np
import time
import FW_1dim as fw


# Parameters
n = 5  # problem size
p = 1.1  # entropy parameter

# Tolerance parameters
delta = 0.001
eps = 0.001
max_iter = 3


# Generate test data
np.random.seed(0)
mu = np.random.randint(0, 10, size=n).astype(float)
nu = np.random.randint(0, 10, size=n).astype(float)

# Cost function: absolute distance
c = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))

# Set M (upper bound for generalized simplex)
M = n * (np.sum(mu) + np.sum(nu))

print("Problem setup:")
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
print("Running Frank-Wolfe...")
start_time = time.time()

# Initialize
xk, x_marg, y_marg, mask1, mask2 = fw.x_init(mu, nu, p, n)
grad_xk = fw.grad(x_marg, y_marg, mask1, mask2, p, c)

# Initialize sum_term for efficient gap calculation
sum_term = np.sum(grad_xk * xk)

# Main iteration loop with detailed output
for k in range(max_iter):
    print(f"\n{'='*80}")
    print(f"ITERATION {k}")
    print(f"{'='*80}")

    # Display transportation plan
    print("\nTransportation Plan (matrix):")
    print(xk)

    # Display gradient
    print("\nGradient (matrix):")
    np.set_printoptions(precision=4, suppress=True)
    print(grad_xk)
    np.set_printoptions()

    # Display marginals
    print("\nMarginals:")
    print(f"  x_marg: {x_marg}")
    print(f"  y_marg: {y_marg}")

    # Search direction
    vk = fw.LMO(xk, grad_xk, M, eps)
    FW_i, FW_j, AFW_i, AFW_j = vk

    print("\nSearch Direction:")
    print(f"  FW indices: ({FW_i}, {FW_j})")
    print(f"  AFW indices: ({AFW_i}, {AFW_j})")

    if FW_i != -1:
        print(f"  FW gradient value: {grad_xk[FW_i, FW_j]:.6f}")
    if AFW_i != -1:
        print(f"  AFW gradient value: {grad_xk[AFW_i, AFW_j]:.6f}")

    # Gap calculation
    gap = fw.gap_calc(grad_xk, vk, M, sum_term)
    print(f"\nGap: {gap:.6f}")

    if (gap <= delta) or (vk == (-1, -1, -1, -1)):
        print(f"\nConverged after {k} iterations")
        break

    # rows and columns update
    rows, cols = set([FW_i, AFW_i]) - {-1}, set([FW_j, AFW_j]) - {-1}
    rows, cols = list(rows), list(cols)

    # Remove contributions before gradient update
    sum_term = fw.update_sum_term(sum_term, grad_xk, xk, mask1, mask2, rows, cols, sign=-1)

    # Calculate stepsize BEFORE applying the step (using CURRENT state)
    if AFW_i != -1:
        gamma0 = xk[AFW_i, AFW_j] - 1e-10
        gammak = fw.step_calc(x_marg, y_marg, grad_xk, mu, nu, vk, c, p, theta=gamma0)
        # Apply Away step
        xk[AFW_i, AFW_j] -= gammak
        x_marg[AFW_i] -= gammak / mu[AFW_i]
        y_marg[AFW_j] -= gammak / nu[AFW_j]
        if FW_i != -1:
            xk[FW_i, FW_j] += gammak
            x_marg[FW_i] += gammak / mu[FW_i]
            y_marg[FW_j] += gammak / nu[FW_j]
    else:
        gamma0 = M - np.sum(xk) + xk[FW_i, FW_j]
        gammak = fw.step_calc(x_marg, y_marg, grad_xk, mu, nu, vk, c, p, theta=gamma0)
        # Apply Frank-Wolfe step
        xk[FW_i, FW_j] += gammak
        x_marg[FW_i] += gammak / mu[FW_i]
        y_marg[FW_j] += gammak / nu[FW_j]

    print(f"\nStepsize (gamma): {gammak:.6f}")

    # Gradient update
    grad_xk = fw.update_grad(x_marg, y_marg, grad_xk, mask1, mask2, c, vk, p)

    # Add back contributions after gradient update
    sum_term = fw.update_sum_term(sum_term, grad_xk, xk, mask1, mask2, rows, cols, sign=+1)
    print(f"Sum term: {sum_term:.6f}")
    print(f"Real sum term: {np.sum(grad_xk * xk):.6f}")

elapsed_time = time.time() - start_time

print(f"\n{'='*80}")
print(f"Frank-Wolfe completed in {elapsed_time:.4f} seconds")
print(f"{'='*80}\n")

# Compute and print final cost
cost = fw.UOT_cost(xk, x_marg, y_marg, c, mu, nu, p)
print(f"Final UOT cost: {cost:.6f}")

# Print additional information
print("\nTransportation plan (matrix):")
print(xk)

print("\nMarginals:")
print(f"  x_marg: {x_marg}")
print(f"  y_marg: {y_marg}")

# Verify constraints
total_mass = np.sum(xk)
print("\nConstraint verification:")
print(f"  Total mass (should be <= M={M:.2f}): {total_mass:.6f}")
print(f"  Sum(x_marg * mu): {np.sum(x_marg * mu):.6f}")
print(f"  Sum(y_marg * nu): {np.sum(y_marg * nu):.6f}")

print("\nExperiment completed.")
