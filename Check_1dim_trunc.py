"""
Experiment: 1D Truncated Frank-Wolfe (vector implementation)
"""

import numpy as np
import time
import FW_1dim_trunc as fw


# Parameters
n = 5  # problem size
p = 0.1  # entropy parameter
R = 2  # truncation radius

# Tolerance parameters
delta = 0.001
eps = 0.001
max_iter = 10


# Generate test data
np.random.seed(0)
mu = np.random.randint(0, 10, size=n).astype(float)
nu = np.random.randint(0, 10, size=n).astype(float)

# Cost function: absolute distance (vector form for truncated support)
# Store only the diagonals within truncation radius
c_vec = []
for k in range(-R + 1, R):
    m = n - abs(k)
    c_vec.extend([abs(k)] * m)
c = np.array(c_vec)

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

# Run the Frank-Wolfe algorithm with detailed iteration output
print("Running Truncated Frank-Wolfe...")
start_time = time.time()

# Initialize
xk, x_marg, y_marg, mask1, mask2 = fw.x_init_trunc(mu, nu, n, c, p)
s_i, s_j = np.zeros(n), np.zeros(n)
grad_xk_x, grad_xk_s = fw.grad_trunc(x_marg, y_marg, mask1, mask2, c, p, n, R)

# Main iteration loop with detailed output
for k in range(max_iter):
    print(f"\n{'='*80}")
    print(f"ITERATION {k}")
    print(f"{'='*80}")

    # Display transportation plan (convert to matrix for readability)
    print("\nTransportation Plan (matrix):")
    xk_matrix = fw.vector_to_matrix(xk, n, R)
    print(xk_matrix)

    # Display gradient
    print("\nGradient for x (matrix):")
    np.set_printoptions(precision=4, suppress=True)
    grad_xk_x_matrix = fw.vector_to_matrix(grad_xk_x, n, R)
    print(grad_xk_x_matrix)
    print("\nGradient for s_i:")
    print(grad_xk_s[0])
    print("Gradient for s_j:")
    print(grad_xk_s[1])
    np.set_printoptions()

    # Display marginals
    print("\nMarginals:")
    print(f"  x_marg: {x_marg}")
    print(f"  y_marg: {y_marg}")

    # Display truncated supports
    print("\nTruncated Supports:")
    print(f"  s_i: {s_i}")
    print(f"  s_j: {s_j}")

    # Search direction
    vk_x = fw.LMO_x(xk, grad_xk_x, M, eps)
    vk_s = fw.LMO_s(s_i, s_j, grad_xk_s, M, eps, mask1, mask2)
    
    FW_x, AFW_x = vk_x
    FW_si, FW_sj, AFW_si, AFW_sj = vk_s

    print("\nSearch Direction for x:")
    if FW_x != -1:
        FW_i, FW_j = fw.vector_index_to_matrix_indices(FW_x, n, R)
        print(f"  FW indices (matrix): ({FW_i}, {FW_j}), vector index: {FW_x}")
        print(f"  FW gradient value: {grad_xk_x[FW_x]:.6f}")
    else:
        print(f"  FW indices: None")
    
    if AFW_x != -1:
        AFW_i, AFW_j = fw.vector_index_to_matrix_indices(AFW_x, n, R)
        print(f"  AFW indices (matrix): ({AFW_i}, {AFW_j}), vector index: {AFW_x}")
        print(f"  AFW gradient value: {grad_xk_x[AFW_x]:.6f}")
    else:
        print(f"  AFW indices: None")

    print("\nSearch Direction for s:")
    print(f"  FW_si: {FW_si}, FW_sj: {FW_sj}")
    print(f"  AFW_si: {AFW_si}, AFW_sj: {AFW_sj}")

    # Gap calculation
    gap = fw.gap_calc_trunc(xk, grad_xk_x, vk_x, M, s_i, s_j, grad_xk_s, vk_s)
    print(f"\nGap: {gap:.6f}")

    if (gap <= delta) or (vk_x == (-1, -1) and vk_s == (-1, -1, -1, -1)):
        print(f"\nConverged after {k} iterations")
        break

    # Apply step update
    xk, x_marg, y_marg, s_i, s_j, v_coords = fw.apply_step_trunc(
        xk, x_marg, y_marg, s_i, s_j, grad_xk_x, grad_xk_s,
        mu, nu, M, vk_x, vk_s, c, p, n, R)

    print(f"\nApplied step - v_coords: {v_coords}")

    # Update gradient
    grad_xk_x, grad_xk_s = fw.update_grad_trunc(x_marg, y_marg, s_i, s_j, grad_xk_x, grad_xk_s, 
                                                 mask1, mask2, p, n, R, v_coords, vk_s)

elapsed_time = time.time() - start_time

print(f"\n{'='*80}")
print(f"Truncated Frank-Wolfe completed in {elapsed_time:.4f} seconds")
print(f"{'='*80}\n")

cost_trunc = fw.truncated_cost(xk, x_marg, y_marg, c, mu, nu, p, s_i, s_j, R)
print(f"Final UOT cost (truncated): {cost_trunc:.6f}")

# Compute and print final costs
cost = fw.UOT_cost_upper(cost_trunc, n, s_i, R, mu)
print(f"Final UOT cost: {cost:.6f}")

# Print additional information
print("\nTransportation plan (matrix):")
xk_matrix = fw.vector_to_matrix(xk, n, R)
print(xk_matrix)

print("\nMarginals:")
print(f"  x_marg: {x_marg}")
print(f"  y_marg: {y_marg}")

print("\nTruncated Supports:")
print(f"  s_i: {s_i}")
print(f"  s_j: {s_j}")

# Verify constraints
total_mass = np.sum(xk)
print("\nConstraint verification:")
print(f"  Total mass (should be <= M={M:.2f}): {total_mass:.6f}")
print(f"  Sum(x_marg * mu): {np.sum(x_marg * mu):.6f}")
print(f"  Sum(y_marg * nu): {np.sum(y_marg * nu):.6f}")
print(f"  Sum(s_i * mu): {np.sum(s_i * mu):.6f}")
print(f"  Sum(s_j * nu): {np.sum(s_j * nu):.6f}")

print("\nExperiment completed.")
