"""
Experiment: 2D Frank-Wolfe with p=2 (9-point stencil implementation)
"""

import numpy as np
import time
from FW_2dim_p2 import (
    x_init_dim2_p2,
    grad_dim2_p2,
    LMO_dim2_p2,
    gap_calc_dim2_p2,
    update_sum_term_dim2_p2,
    apply_step_dim2_p2,
    opt_step_dim2_p2,
    grad_update_dim2_p2,
    cost_dim2_p2,
    to_dense_dim2_p2,
)


# Parameters
n = 3  # grid size (n x n)
p = 2  # entropy parameter (p=2)

# Tolerance parameters
delta = 0.001
eps = 0.001
max_iter = 20


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
x, x_marg, y_marg, mask1, mask2 = x_init_dim2_p2(mu, nu, n)
grad = grad_dim2_p2(x_marg, y_marg, mask1, mask2, n)

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
    print(f"  [0] Same (dist=0):     sum={np.sum(x[0]):.4f}")
    print(f"  [1] Up (dist=1):       sum={np.sum(x[1]):.4f}")
    print(f"  [2] Left (dist=1):     sum={np.sum(x[2]):.4f}")
    print(f"  [3] Down (dist=1):     sum={np.sum(x[3]):.4f}")
    print(f"  [4] Right (dist=1):    sum={np.sum(x[4]):.4f}")
    print(f"  [5] Up-left (√2):      sum={np.sum(x[5]):.4f}")
    print(f"  [6] Down-left (√2):    sum={np.sum(x[6]):.4f}")
    print(f"  [7] Up-right (√2):     sum={np.sum(x[7]):.4f}")
    print(f"  [8] Down-right (√2):   sum={np.sum(x[8]):.4f}")
    print(f"  Total mass: {np.sum(x):.4f}")
    
    # Dense transportation plan
    x_dense = to_dense_dim2_p2(x, n)
    print("\nDense transportation plan x_dense (n, n, n, n):")
    print(x_dense)
    
    # Display marginals
    print(f"\nMarginals:")
    print(f"  x_marg (relative):")
    print(x_marg)
    print(f"  y_marg (relative):")
    print(y_marg)
    print(f"  x_marg * mu (absolute):")
    print(x_marg * mu)
    print(f"  y_marg * nu (absolute):")
    print(y_marg * nu)
    
    # Display gradient statistics
    print(f"\nGradient statistics:")
    print(f"  min(grad) = {np.min(grad):.6f}")
    print(f"  max(grad) = {np.max(grad):.6f}")
    
    # Dense gradient tensor (same stencil-to-dense mapping)
    grad_dense = to_dense_dim2_p2(grad, n)
    print("\nDense gradient tensor grad_dense (n, n, n, n):")
    print(grad_dense)
    
    # Get search directions using LMO
    (FW_compact, FW_full), (AFW_compact, AFW_full) = LMO_dim2_p2(x, grad, M, eps)
    
    mat_FW, i_FW, j_FW = FW_compact
    mat_AFW, i_AFW, j_AFW = AFW_compact
    x1FW, x2FW, y1FW, y2FW = FW_full
    x1AFW, x2AFW, y1AFW, y2AFW = AFW_full
    
    print(f"\nSearch Direction:")
    print(f"  FW (Frank-Wolfe):")
    if x1FW != -1:
        print(f"    Compact: mat_idx={mat_FW}, source=({i_FW},{j_FW})")
        print(f"    Full: ({x1FW},{x2FW}) → ({y1FW},{y2FW})")
        print(f"    Gradient value: {grad[mat_FW, i_FW, j_FW]:.6f}")
        print(f"    Current mass: {x[mat_FW, i_FW, j_FW]:.6f}")
    else:
        print(f"    None (no descent direction)")
    
    print(f"  AFW (Away Frank-Wolfe):")
    if x1AFW != -1:
        print(f"    Compact: mat_idx={mat_AFW}, source=({i_AFW},{j_AFW})")
        print(f"    Full: ({x1AFW},{x2AFW}) → ({y1AFW},{y2AFW})")
        print(f"    Gradient value: {grad[mat_AFW, i_AFW, j_AFW]:.6f}")
        print(f"    Current mass: {x[mat_AFW, i_AFW, j_AFW]:.6f}")
    else:
        print(f"    None (no away direction)")
    
    # Gap calculation
    gap = gap_calc_dim2_p2(grad, FW_compact, M, sum_term)
    print(f"\nGap: {gap:.6f}")
    
    # Check convergence
    if gap <= delta:
        print(f"\n✓ Converged after {k} iterations (gap <= delta)")
        break
    
    if x1FW == -1 and x1AFW == -1:
        print(f"\n✓ Converged after {k} iterations (no directions found)")
        break
    
    # Subtract old contributions from sum_term
    sum_term = update_sum_term_dim2_p2(sum_term, grad, x, FW_full, AFW_full, n, -1)
    
    # Apply step update
    x, x_marg, y_marg = apply_step_dim2_p2(
        x, x_marg, y_marg, mu, nu, M,
        FW_compact, FW_full, AFW_compact, AFW_full
    )
    
    # Compute step size for display (already computed in apply_step)
    mat_idx = (mat_FW, mat_AFW)
    full = (FW_full, AFW_full)
    gammak_opt = opt_step_dim2_p2(x_marg, y_marg, mu, nu, mat_idx, full)
    
    if x1AFW != -1:
        gamma0 = x[mat_AFW, x1AFW, x2AFW] + apply_step_dim2_p2.__code__.co_consts[0]  # get 1e-10
        gammak = min(gammak_opt, gamma0)
    else:
        gamma0 = M - np.sum(x) + x[mat_FW, x1FW, x2FW]
        gammak = min(gammak_opt, gamma0)
    
    print(f"\nStepsize:")
    print(f"  Optimal: {gammak_opt:.6f}")
    print(f"  Max allowed (gamma0): {gamma0:.6f}")
    print(f"  Used (gammak): {gammak:.6f}")
    
    # Update gradient
    grad = grad_update_dim2_p2(x_marg, y_marg, grad, mask1, mask2, 
                                            FW_full, AFW_full)
    
    # Add new contributions to sum_term
    sum_term = update_sum_term_dim2_p2(sum_term, grad, x, FW_full, AFW_full, n, +1)
    
    print(f"\nSum term:")
    print(f"  Maintained: {sum_term:.6f}")
    print(f"  Recomputed: {np.sum(grad * x):.6f}")
    print(f"  Difference: {abs(sum_term - np.sum(grad * x)):.2e}")

elapsed_time = time.time() - start_time

print(f"\n{'='*80}")
print(f"Frank-Wolfe completed in {elapsed_time:.4f} seconds")
print(f"{'='*80}\n")

# Compute and print final cost
cost = cost_dim2_p2(x, x_marg, y_marg, mu, nu)
print(f"Final UOT cost: {cost:.6f}")

# Print transportation plan summary
print(f"\nFinal Transportation Plan Summary:")
print(f"  Total mass: {np.sum(x):.6f}")
for mat_idx in range(9):
    mass = np.sum(x[mat_idx])
    if mass > 1e-10:
        print(f"  Matrix {mat_idx}: {mass:.6f}")

print(f"\nFinal Marginals:")
print(f"  x_marg * mu:")
print(x_marg * mu)
print(f"  y_marg * nu:")
print(y_marg * nu)

# Verify constraints
total_mass = np.sum(x)
print(f"\nConstraint verification:")
print(f"  Total mass (should be <= M={M:.2f}): {total_mass:.6f}")
print(f"  Sum(x_marg * mu): {np.sum(x_marg * mu):.6f}")
print(f"  Sum(y_marg * nu): {np.sum(y_marg * nu):.6f}")
print(f"  Conservation error: {abs(np.sum(x_marg * mu) - np.sum(y_marg * nu)):.2e}")

# Check if any entries are negative
if np.any(x < -1e-10):
    print(f"\n WARNING: Negative entries found in x!")
    neg_entries = np.where(x < -1e-10)
    for mat_idx, i, j in zip(*neg_entries):
        print(f"    x[{mat_idx}, {i}, {j}] = {x[mat_idx, i, j]:.6f}")

print("\n✓ Experiment completed.")