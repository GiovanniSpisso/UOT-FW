"""
Experiment: 1D Frank-Wolfe with p=2 (Array-based 3n vector implementation)
"""

import numpy as np
import time
import FW_1dim_p2 as fw_p2


# Parameters
n = 5  # problem size
p = 2  # entropy parameter (p=2)

# Tolerance parameters
delta = 0.001
eps = 0.001
max_iter = 10


# Generate test data
np.random.seed(0)
mu = np.random.randint(1, 10, size=n).astype(float)
nu = np.random.randint(1, 10, size=n).astype(float)

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
print("Running Frank-Wolfe with p=2...")
start_time = time.time()

# Initialize
xk, x_marg, y_marg = fw_p2.x_init_p2(mu, nu, n)
grad_xk = fw_p2.grad_p2(x_marg, y_marg, n)

# Initialize sum_term for efficient gap calculation
sum_term = np.dot(grad_xk, xk)

# Main iteration loop with detailed output
for k in range(max_iter):
    print(f"\n{'='*80}")
    print(f"ITERATION {k}")
    print(f"{'='*80}")
    
    # Display transportation plan as matrix
    print("\nTransportation Plan (3n vector):")
    print(f"  Upper diagonal: {xk[:n]}")
    print(f"  Main diagonal:  {xk[n:2*n]}")
    print(f"  Lower diagonal: {xk[2*n:3*n]}")
    
    # Convert to matrix for display
    xk_matrix = fw_p2.vec_to_mat_p2(xk, n)
    print("\nTransportation Plan (as matrix):")
    print(xk_matrix)
    
    # Display gradient as matrix
    grad_matrix = fw_p2.vec_to_mat_p2(grad_xk, n)
    print("\nGradient (as matrix):")
    np.set_printoptions(precision=4, suppress=True)
    print(grad_matrix)
    np.set_printoptions()  # Reset to default
    
    # Display marginals
    print(f"\nMarginals:")
    print(f"  x_marg: {x_marg}")
    print(f"  y_marg: {y_marg}")
    
    # Search direction
    vk = fw_p2.LMO_p2(xk, grad_xk, M, eps)
    FW, AFW = vk
    
    print(f"\nSearch Direction:")
    print(f"  FW (Frank-Wolfe) index (3n): {FW}")
    print(f"  AFW (Away FW) index (3n): {AFW}")
    
    if FW != -1:
        result = fw_p2.vec_i_to_mat_i_p2(FW, n)
        if result is not None:
            FW_i, FW_j = result
            print(f"  FW matrix coordinates: ({FW_i}, {FW_j}), gradient value: {grad_xk[FW]:.6f}")
    
    if AFW != -1:
        result = fw_p2.vec_i_to_mat_i_p2(AFW, n)
        if result is not None:
            AFW_i, AFW_j = result
            print(f"  AFW matrix coordinates: ({AFW_i}, {AFW_j}), gradient value: {grad_xk[AFW]:.6f}")
    
    # Gap calculation
    gap = fw_p2.gap_calc_p2(grad_xk, vk, M, sum_term)
    print(f"\nGap: {gap:.6f}")
    
    if (gap <= delta) or (vk == (-1, -1)):
        print(f"\nConverged after {k} iterations")
        break
    
    # Convert 3n indices to matrix coordinates
    FW_i, FW_j = (-1, -1)
    AFW_i, AFW_j = (-1, -1)
    
    if FW != -1:
        result = fw_p2.vec_i_to_mat_i_p2(FW, n)
        if result is not None:
            FW_i, FW_j = result
    
    if AFW != -1:
        result = fw_p2.vec_i_to_mat_i_p2(AFW, n)
        if result is not None:
            AFW_i, AFW_j = result
    
    # Remove contributions before gradient update
    sum_term = fw_p2.update_sum_term_p2(sum_term, grad_xk, xk, vk, n, sign=-1)
    
    # Calculate stepsize BEFORE applying the step (using CURRENT state)
    if AFW_i != -1:
        gamma0 = xk[AFW] - 1e-10
        gammak = min(fw_p2.opt_step(x_marg, y_marg, mu, nu,
                          coords=(FW_i, FW_j, AFW_i, AFW_j)), gamma0)
        # Apply Away step
        xk[AFW] -= gammak
        x_marg[AFW_i] -= gammak / mu[AFW_i]
        y_marg[AFW_j] -= gammak / nu[AFW_j]
        if FW_i != -1:
            xk[FW] += gammak
            x_marg[FW_i] += gammak / mu[FW_i]
            y_marg[FW_j] += gammak / nu[FW_j]
    else:
        gamma0 = M - np.sum(xk) + xk[FW]
        gammak = min(fw_p2.opt_step(x_marg, y_marg, mu, nu,
                          coords=(FW_i, FW_j, AFW_i, AFW_j)), gamma0)
        # Apply Frank-Wolfe step
        xk[FW] += gammak
        x_marg[FW_i] += gammak / mu[FW_i]
        y_marg[FW_j] += gammak / nu[FW_j]
    
    print(f"\nStepsize (gamma): {gammak:.6f}")
    
    # Gradient update
    grad_xk = fw_p2.update_grad_p2(x_marg, y_marg, grad_xk,
                                     coords=(FW_i, FW_j, AFW_i, AFW_j))
    
    # Add back contributions after gradient update
    sum_term = fw_p2.update_sum_term_p2(sum_term, grad_xk, xk, vk, n, sign=+1)
    print(f"Sum term: {sum_term:.6f}")
    print(f"Real sum term: {np.sum(grad_xk * xk):.6f}")

elapsed_time = time.time() - start_time

print(f"\n{'='*80}")
print(f"Frank-Wolfe completed in {elapsed_time:.4f} seconds")
print(f"{'='*80}\n")

# Compute and print final cost
cost = fw_p2.cost_p2(xk, x_marg, y_marg, mu, nu)
print(f"Final UOT cost: {cost:.6f}")

# Print additional information
print(f"\nTransportation plan (3n vector):")
print(f"  Upper diagonal: {xk[:n]}")
print(f"  Main diagonal: {xk[n:2*n]}")
print(f"  Lower diagonal: {xk[2*n:3*n]}")

print(f"\nMarginals:")
print(f"  x_marg: {x_marg*mu}")
print(f"  y_marg: {y_marg*nu}")

# Verify constraints
total_mass = np.sum(xk)
print(f"\nConstraint verification:")
print(f"  Total mass (should be <= M={M:.2f}): {total_mass:.6f}")
print(f"  Sum(x_marg * mu): {np.sum(x_marg * mu):.6f}")
print(f"  Sum(y_marg * nu): {np.sum(y_marg * nu):.6f}")

print("\nExperiment completed.")
