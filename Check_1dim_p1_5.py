"""
Experiment: 1D Frank-Wolfe with p=1.5 (Array-based 7n vector implementation)
"""

import numpy as np
import time
import FW_1dim_p1_5 as fw_p1_5


# Parameters
n = 5  # problem size
p = 1.5  # entropy parameter (p=1.5)

# Tolerance parameters
delta = 0.001
eps = 0.001
max_iter = 10


# Generate test data
np.random.seed(0)
mu = np.random.randint(0, 10, size=n).astype(float)
nu = np.random.randint(0, 10, size=n).astype(float)

# Set M (upper bound for generalized simplex)
M = 2 * (np.sum(mu) + np.sum(nu))

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

# Mask zero entries in mu and nu to deal with measures with zero mass
mu = np.ma.masked_equal(mu, 0)
nu = np.ma.masked_equal(nu, 0)

# Run the Frank-Wolfe algorithm with detailed iteration output
print("Running Frank-Wolfe with p=1.5...")
start_time = time.time()

# Initialize
xk, x_marg, y_marg, mask_7n = fw_p1_5.x_init_p1_5(mu, nu, n)
grad_xk = fw_p1_5.grad_p1_5(x_marg, y_marg, n, mask_7n)

# Initialize sum_term for efficient gap calculation
sum_term = np.sum(grad_xk * xk)

# Main iteration loop with detailed output
for k in range(max_iter):
    print(f"\n{'='*80}")
    print(f"ITERATION {k}")
    print(f"{'='*80}")
    
    # Display transportation plan as vector
    print("\nTransportation Plan (7n vector):")
    print(f"  Upper3 diagonal: {xk[:n]}")
    print(f"  Upper2 diagonal: {xk[n:2*n]}")
    print(f"  Upper1 diagonal: {xk[2*n:3*n]}")
    print(f"  Main diagonal:   {xk[3*n:4*n]}")
    print(f"  Lower1 diagonal: {xk[4*n:5*n]}")
    print(f"  Lower2 diagonal: {xk[5*n:6*n]}")
    print(f"  Lower3 diagonal: {xk[6*n:7*n]}")
    
    # Convert to matrix for display
    xk_matrix = fw_p1_5.vec_to_mat_p1_5(xk, n)
    print("\nTransportation Plan (as matrix):")
    print(xk_matrix)
    
    # Display gradient as matrix
    grad_matrix = fw_p1_5.vec_to_mat_p1_5(grad_xk, n)
    print("\nGradient (as matrix):")
    np.set_printoptions(precision=4, suppress=True)
    print(grad_matrix)
    np.set_printoptions()  # Reset to default
    
    # Display marginals
    print(f"\nMarginals:")
    print(f"  x_marg: {x_marg}")
    print(f"  y_marg: {y_marg}")
    
    # Search direction
    vk = fw_p1_5.LMO_p1_5(xk, grad_xk, M, eps)
    FW, AFW = vk
    
    print(f"\nSearch Direction:")
    print(f"  FW (Frank-Wolfe) index (7n): {FW}")
    print(f"  AFW (Away FW) index (7n): {AFW}")
    
    if FW != -1:
        result = fw_p1_5.vec_i_to_mat_i_p1_5(FW, n)
        if result is not None:
            FW_i, FW_j = result
            print(f"  FW matrix coordinates: ({FW_i}, {FW_j}), gradient value: {grad_xk[FW]:.6f}")
    
    if AFW != -1:
        result = fw_p1_5.vec_i_to_mat_i_p1_5(AFW, n)
        if result is not None:
            AFW_i, AFW_j = result
            print(f"  AFW matrix coordinates: ({AFW_i}, {AFW_j}), gradient value: {grad_xk[AFW]:.6f}")
    
    # Gap calculation
    gap = fw_p1_5.gap_calc_p1_5(grad_xk, vk, M, sum_term)
    print(f"\nGap: {gap:.6f}")
    
    if (gap <= delta) or (vk == (-1, -1)):
        print(f"\nConverged after {k} iterations")
        break
    
    # Convert 7n indices to matrix coordinates
    FW_i, FW_j = (-1, -1)
    AFW_i, AFW_j = (-1, -1)
    
    if FW != -1:
        result = fw_p1_5.vec_i_to_mat_i_p1_5(FW, n)
        if result is not None:
            FW_i, FW_j = result
    
    if AFW != -1:
        result = fw_p1_5.vec_i_to_mat_i_p1_5(AFW, n)
        if result is not None:
            AFW_i, AFW_j = result
    
    # Remove contributions before gradient update
    sum_term = fw_p1_5.update_sum_term_p1_5(sum_term, grad_xk, xk, (FW_i, FW_j, AFW_i, AFW_j), n, sign=-1)
    
    # Apply step update
    xk, x_marg, y_marg = fw_p1_5.apply_step_p1_5(
        xk, x_marg, y_marg, grad_xk, mu, nu, M, vk,
        coords=(FW_i, FW_j, AFW_i, AFW_j)
    )
    
    # Calculate stepsize (for informational purposes)
    if AFW_i != -1:
        gammak = xk[AFW]  # Informational only
        print(f"\nStep type: Away Frank-Wolfe")
    else:
        gammak = xk[FW]  # Informational only
        print(f"\nStep type: Frank-Wolfe")
    
    print(f"Stepsize (gamma): {gammak:.6f}")
    
    # Gradient update
    grad_xk = fw_p1_5.update_grad_p1_5(x_marg, y_marg, grad_xk, coords=(FW_i, FW_j, AFW_i, AFW_j))
    
    # Add back contributions after gradient update
    sum_term = fw_p1_5.update_sum_term_p1_5(sum_term, grad_xk, xk, (FW_i, FW_j, AFW_i, AFW_j), n, sign=+1)
    print(f"Sum term: {sum_term:.6f}")
    print(f"Real sum term: {np.sum(grad_xk * xk):.6f}")

elapsed_time = time.time() - start_time

print(f"\n{'='*80}")
print(f"Frank-Wolfe completed in {elapsed_time:.4f} seconds")
print(f"{'='*80}\n")

# Compute and print final cost
cost = fw_p1_5.cost_p1_5(xk, x_marg, y_marg, mu, nu)
print(f"Final UOT cost: {cost:.6f}")

# Print additional information
print(f"\nTransportation plan (7n vector):")
print(f"  Upper3 diagonal: {xk[:n]}")
print(f"  Upper2 diagonal: {xk[n:2*n]}")
print(f"  Upper1 diagonal: {xk[2*n:3*n]}")
print(f"  Main diagonal: {xk[3*n:4*n]}")
print(f"  Lower1 diagonal: {xk[4*n:5*n]}")
print(f"  Lower2 diagonal: {xk[5*n:6*n]}")
print(f"  Lower3 diagonal: {xk[6*n:7*n]}")

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
