"""
Experiment: 2D Frank-Wolfe with p=2 (9-point stencil implementation)
"""

import numpy as np
import time
import FW_2dim_p2 as fw2d


# Parameters
n = 3  # grid size (n x n)
p = 2  # entropy parameter (p=2)

# Tolerance parameters
delta = 0.001
eps = 0.001
max_iter = 5


# Generate test data
np.random.seed(0)
mu = np.random.randint(0, 11, size=(n,n)).astype(float)
nu = np.random.randint(0, 11, size=(n,n)).astype(float)

# Alternative: simple test case
# mu = np.array([[0.5, 2], [0.5, 2]])
# nu = np.array([[1, 0.1], [1, 0.1]])

# Set M (upper bound for generalized simplex)
M = n * n * (np.sum(mu) + np.sum(nu))

print(f"Problem setup:")
print(f"  Grid size: {n} x {n}")
print(f"  p = {p}")
print(f"  M (upper bound) = {M:.2f}")
print(f"  max_iter = {max_iter}")
print(f"  delta (convergence tol) = {delta}")
print(f"  eps (direction tol) = {eps}")
print(f"\nMeasures:")
print(f"  mu =")
print(mu)
print(f"  nu =")
print(nu)
print(f"  sum(mu) = {np.sum(mu):.2f}")
print(f"  sum(nu) = {np.sum(nu):.2f}")
print()

# Run the Frank-Wolfe algorithm with detailed iteration output
print("Running 2D Frank-Wolfe with p=2...")
start_time = time.time()

# Initialize
x, x_marg, y_marg, mask1, mask2 = fw2d.x_init_dim2_p2(mu, nu, n)
grad = fw2d.grad_dim2_p2(x_marg, y_marg, mask1, mask2, n)

# Initialize sum_term for efficient gap calculation
sum_term = np.sum(grad * x)

# Print formatting options
np.set_printoptions(precision=4, suppress=True, linewidth=100)

# Main iteration loop with detailed output
for k in range(max_iter):
    print(f"\n{'='*80}")
    print(f"ITERATION {k}")
    print(f"{'='*80}")
    
    # Display transportation plan
    print("\nTransportation Plan (9, n, n) compact format:")
    for mat_idx in range(9):
        print(f"  Matrix {mat_idx}: sum={np.sum(x[mat_idx]):.4f}")
    print(f"  Total mass: {np.sum(x):.4f}")
    
    print("\nTransportation Plan (compact):")
    print(x)
    
    # Display marginals
    print(f"\nMarginals:")
    print(f"  x_marg: {x_marg}")
    print(f"  y_marg: {y_marg}")
    
    # Display gradient
    print("\nGradient (compact):")
    print(grad)
    
    # Get search directions
    (FW_compact, FW_full), (AFW_compact, AFW_full) = fw2d.LMO_dim2_p2(x, grad, M, eps)
    
    mat_FW, i_FW, j_FW = FW_compact
    mat_AFW, i_AFW, j_AFW = AFW_compact
    x1FW, x2FW, y1FW, y2FW = FW_full
    x1AFW, x2AFW, y1AFW, y2AFW = AFW_full
    
    print(f"\nSearch Direction:")
    print(f"  FW (compact): mat_idx={mat_FW}, ({i_FW},{j_FW})")
    print(f"  FW (full): ({x1FW},{x2FW}) → ({y1FW},{y2FW})")
    print(f"  AFW (compact): mat_idx={mat_AFW}, ({i_AFW},{j_AFW})")
    print(f"  AFW (full): ({x1AFW},{x2AFW}) → ({y1AFW},{y2AFW})")
    
    if x1FW != -1:
        print(f"  FW gradient value: {grad[mat_FW, i_FW, j_FW]:.6f}")
    
    if x1AFW != -1:
        print(f"  AFW gradient value: {grad[mat_AFW, i_AFW, j_AFW]:.6f}")
    
    # Gap calculation
    gap = fw2d.gap_calc_dim2_p2(grad, FW_compact, M, sum_term)
    print(f"\nGap: {gap:.6f}")
    
    if (gap <= delta) or ((x1FW, x1AFW) == (-1, -1)):
        print(f"\nConverged after {k} iterations")
        break
    
    # Remove contributions before update
    sum_term = fw2d.update_sum_term_dim2_p2(sum_term, grad, x, FW_full, AFW_full, n, sign=-1)
    
    # Calculate stepsize BEFORE applying the step (using CURRENT state)
    if x1AFW != -1:
        gamma0 = x[mat_AFW, x1AFW, x2AFW] - 1e-10
        gammak = min(fw2d.opt_step_dim2_p2(x_marg, y_marg, mu, nu,
                                            mat_idx=(mat_FW, mat_AFW),
                                            full=(FW_full, AFW_full)), gamma0)
        # Apply Away step
        x[mat_AFW, x1AFW, x2AFW] -= gammak
        x_marg[x1AFW, x2AFW] -= gammak / mu[x1AFW, x2AFW]
        y_marg[y1AFW, y2AFW] -= gammak / nu[y1AFW, y2AFW]
        if x1FW != -1:
            x[mat_FW, x1FW, x2FW] += gammak
            x_marg[x1FW, x2FW] += gammak / mu[x1FW, x2FW]
            y_marg[y1FW, y2FW] += gammak / nu[y1FW, y2FW]
    else:
        gamma0 = M - np.sum(x) + x[mat_FW, x1FW, x2FW]
        gammak = min(fw2d.opt_step_dim2_p2(x_marg, y_marg, mu, nu,
                                            mat_idx=(mat_FW, mat_AFW),
                                            full=(FW_full, AFW_full)), gamma0)
        # Apply Frank-Wolfe step
        x[mat_FW, x1FW, x2FW] += gammak
        x_marg[x1FW, x2FW] += gammak / mu[x1FW, x2FW]
        y_marg[y1FW, y2FW] += gammak / nu[y1FW, y2FW]
    
    print(f"\nStepsize (gamma): {gammak:.6f}")
    
    # Gradient update
    grad = fw2d.grad_update_dim2_p2(x_marg, y_marg, grad, mask1, mask2,
                                     FW_full, AFW_full)
    
    # Add back contributions after gradient update
    sum_term = fw2d.update_sum_term_dim2_p2(sum_term, grad, x, FW_full, AFW_full, n, sign=+1)
    print(f"Sum term: {sum_term:.6f}")
    print(f"Real sum term: {np.sum(grad * x):.6f}")

elapsed_time = time.time() - start_time

print(f"\n{'='*80}")
print(f"Frank-Wolfe completed in {elapsed_time:.4f} seconds")
print(f"{'='*80}\n")

# Compute and print final cost
cost = fw2d.cost_dim2_p2(x, x_marg, y_marg, mu, nu)
print(f"Final UOT cost: {cost:.6f}")

# Print additional information
print(f"\nTransportation plan summary:")
for mat_idx in range(9):
    mass = np.sum(x[mat_idx])
    if mass > 1e-10:
        print(f"  Matrix {mat_idx}: {mass:.6f}")

print(f"\nMarginals:")
print(f"  x_marg * mu: {x_marg * mu}")
print(f"  y_marg * nu: {y_marg * nu}")

# Verify constraints
total_mass = np.sum(x)
print(f"\nConstraint verification:")
print(f"  Total mass (should be <= M={M:.2f}): {total_mass:.6f}")
print(f"  Sum(x_marg * mu): {np.sum(x_marg * mu):.6f}")
print(f"  Sum(y_marg * nu): {np.sum(y_marg * nu):.6f}")

print("\nExperiment completed.")