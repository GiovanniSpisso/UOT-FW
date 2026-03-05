import time
import os
import numpy as np
import matplotlib.pyplot as plt
from FW_2dim import x_init_dim2, grad_dim2, LMO_dim2, gap_calc_dim2, \
    grad_update_dim2, UOT_cost, apply_step_dim2, update_sum_term_dim2
from FW_2dim_p2 import x_init_dim2_p2, grad_dim2_p2, LMO_dim2_p2, gap_calc_dim2_p2, \
    grad_update_dim2_p2, apply_step_dim2_p2, update_sum_term_dim2_p2, cost_dim2_p2, to_dense_dim2_p2
from FW_2dim_trunc import PW_FW_dim2_trunc, truncated_cost_dim2, cost_matrix_trunc_dim2, \
    x_init_trunc_dim2, grad_trunc_dim2, LMO_trunc_dim2_x, LMO_trunc_dim2_s, \
    gap_calc_trunc_dim2, update_grad_trunc_dim2, apply_step_trunc_dim2

# Set seed for replicability
np.random.seed(0)
# Grid size
n = 100  
# Define two positive and discrete measures
mu = np.random.randint(0, 101, size=(n, n))
nu = np.random.randint(0, 101, size=(n, n))

# Main parameters to set
p = 2                                                     # power of the entropy
M = 2 * (np.sum(mu) + np.sum(nu))                         # upper bound for delimiting the generalized simplex
R = 2                                                     # truncation radius
max_iter = 2000
sample_freq = max(1, max_iter // 10)                      # sample 10% of iterations
delta = 0.001                                             # tolerance to stop the gap
eps = 0.001                                               # tolerance for calculating the descent direction

# Build cost matrix for dense case (2D Euclidean)
idx = np.arange(n)
di = (idx[:, None] - idx[None, :]) ** 2
dj = (idx[:, None] - idx[None, :]) ** 2
c = np.sqrt(di[:, None, :, None] + dj[None, :, None, :])

###########################################################
###################     VANILLA FW     ####################
###########################################################
print("\n" + "="*60)
print("Computing Cost VS Time for Vanilla FW")
print("="*60)

# initial transportation plan, marginals and gradient initialization
cost_list = []
time_list = []
start_time = time.time()
no_time = 0

xk, x_marg, y_marg, mask1, mask2 = x_init_dim2(mu, nu, p, n)
grad_xk = grad_dim2(x_marg, y_marg, mask1, mask2, p, c)

# Initialize sum_term for efficient gap calculation
sum_term = np.sum(grad_xk * xk)
k = 0
for k in range(max_iter):
    if k % sample_freq == 0:
        tot_time = time.time()
        time_list.append(tot_time - no_time - start_time)
        cost_list.append(UOT_cost(xk, x_marg, y_marg, c, mu, nu, p))
        no_time += time.time() - tot_time

    # search direction
    vk = LMO_dim2(xk, grad_xk, M, eps)

    # gap calculation
    gap = gap_calc_dim2(grad_xk, vk, M, sum_term)

    if (gap <= delta) or (vk == ((-1,-1,-1,-1),(-1,-1,-1,-1))):
      print("Converged after: ", k, " iterations.")
      break

    # coordinates + rows and columns update
    (x1FW, x2FW, y1FW, y2FW), (x1AFW, x2AFW, y1AFW, y2AFW) = vk

    # Collect affected target and source coordinates
    target_coords = {(y1FW, y2FW), (y1AFW, y2AFW)} - {(-1, -1)}
    source_coords = {(x1FW, x2FW), (x1AFW, x2AFW)} - {(-1, -1)}

    # Remove contributions from affected coordinates before gradient update
    sum_term = update_sum_term_dim2(sum_term, grad_xk, xk, mask1, mask2, source_coords, target_coords, sign=-1)
    
    # Apply step update
    xk, x_marg, y_marg = apply_step_dim2(xk, x_marg, y_marg, grad_xk, mu, nu, M, vk, c, p)

    # gradient update
    grad_xk = grad_update_dim2(x_marg, y_marg, grad_xk, mask1, mask2, c, vk, p)
    
    # Add back contributions from affected coordinates after gradient update
    sum_term = update_sum_term_dim2(sum_term, grad_xk, xk, mask1, mask2, source_coords, target_coords, sign=1)

# Record final cost if not already recorded
if k % sample_freq != 0:
    tot_time = time.time()
    time_list.append(tot_time - no_time - start_time)
    cost_list.append(UOT_cost(xk, x_marg, y_marg, c, mu, nu, p))

cost_FW = np.array(cost_list)
time_FW = np.array(time_list)



###########################################################
###################     FW for p=2     ####################
###########################################################
print("\n" + "="*60)
print("Computing Cost VS Time for FW for p=2")
print("="*60)

# transportation plan, marginals and gradient initialization
cost_list = []
time_list = []
start_time = time.time()
no_time = 0

xk, x_marg, y_marg, mask1, mask2 = x_init_dim2_p2(mu, nu, n)
grad_xk = grad_dim2_p2(x_marg, y_marg, mask1, mask2, n)

# Initialize sum_term for efficient gap calculation
sum_term = np.sum(grad_xk * xk)

for k in range(max_iter):
    if k % sample_freq == 0:
        tot_time = time.time()
        time_list.append(tot_time - no_time - start_time)
        cost_list.append(cost_dim2_p2(xk, x_marg, y_marg, mu, nu))
        no_time += time.time() - tot_time

    # search direction (returns compact and full indices)
    (comp_FW, full_FW), (comp_AFW, full_AFW) = LMO_dim2_p2(xk, grad_xk, M, eps)

    # gap calculation
    gap = gap_calc_dim2_p2(grad_xk, comp_FW, M, sum_term)

    if (gap <= delta) or (full_FW == (-1, -1, -1, -1) and full_AFW == (-1, -1, -1, -1)):
      print("Converged after: ", k, " iterations ")
      break

    # Remove contributions from affected coordinates before gradient update
    sum_term = update_sum_term_dim2_p2(sum_term, grad_xk, xk, full_FW, full_AFW, n, sign=-1)
    
    # Apply step update
    xk, x_marg, y_marg = apply_step_dim2_p2(xk, x_marg, y_marg, mu, nu, M, comp_FW, full_FW, comp_AFW, full_AFW)

    # gradient update
    grad_xk = grad_update_dim2_p2(x_marg, y_marg, grad_xk, mask1, mask2, full_FW, full_AFW)
    
    # Add back contributions from affected coordinates after gradient update
    sum_term = update_sum_term_dim2_p2(sum_term, grad_xk, xk, full_FW, full_AFW, n, sign=1)

# Record final cost if not already recorded
if k % 100 != 0:
    tot_time = time.time()
    time_list.append(tot_time - no_time - start_time)
    cost_list.append(cost_dim2_p2(xk, x_marg, y_marg, mu, nu))

cost_FW_p2 = np.array(cost_list)
time_FW_p2 = np.array(time_list)



###########################################################
################     FW Truncated     #####################
###########################################################
print("\n" + "="*60)
print("Computing Cost VS Time for FW Truncated")
print("="*60)

# Create truncated cost matrix
c_trunc, displacement_map = cost_matrix_trunc_dim2(R)

# Run truncated FW with time tracking
cost_list = []
time_list = []
start_time = time.time()

# Manual iteration with time tracking
xk, x_marg, y_marg, mask1, mask2 = x_init_trunc_dim2(mu, nu, n, R, p)
s_i, s_j = np.zeros((n, n)), np.zeros((n, n))
grad_xk_x, grad_xk_s = grad_trunc_dim2(x_marg, y_marg, mask1, mask2, c_trunc, displacement_map, p, n, R)

no_time = 0

for k in range(max_iter):
    if k % sample_freq == 0:
        tot_time = time.time()
        time_list.append(tot_time - no_time - start_time)
        cost_list.append(truncated_cost_dim2(xk, x_marg, y_marg, c_trunc, mu, nu, p, s_i, s_j, R))
        no_time += time.time() - tot_time
    
    # LMO calls
    (i_FW, i_AFW) = LMO_trunc_dim2_x(xk, grad_xk_x, displacement_map, M, eps)
    vk_x = (i_FW, i_AFW)
    FW_si, FW_sj, AFW_si, AFW_sj = LMO_trunc_dim2_s(s_i, s_j, grad_xk_s, M, eps, mask1, mask2)
    vk_s = (FW_si, FW_sj, AFW_si, AFW_sj)

    # gap calculation
    gap = gap_calc_trunc_dim2(xk, grad_xk_x, i_FW[0], M, s_i, s_j, grad_xk_s, (FW_si, FW_sj))

    if (gap <= delta) or (i_FW[0][0] == -1 and i_AFW[0][0] == -1):
        print("Converged after: ", k, " iterations ")
        break
    
    # Apply step update
    xk, x_marg, y_marg, s_i, s_j = apply_step_trunc_dim2(xk, x_marg, y_marg, s_i, s_j, 
                                                         grad_xk_x, grad_xk_s, mu, nu, 
                                                         M, vk_x, vk_s, c_trunc, p, R)

    # Update gradient
    grad_xk_x, grad_xk_s = update_grad_trunc_dim2(x_marg, y_marg, s_i, s_j, grad_xk_x, grad_xk_s, 
                                                  mask1, mask2, c_trunc, displacement_map, p, 
                                                  R, vk_x, vk_s)

# Record final cost if not already recorded
if k % sample_freq != 0:
    tot_time = time.time()
    time_list.append(tot_time - no_time - start_time)
    cost_list.append(truncated_cost_dim2(xk, x_marg, y_marg, c_trunc, mu, nu, p, s_i, s_j, R))

cost_FW_trunc = np.array(cost_list)
time_FW_trunc = np.array(time_list)



###########################################################
######################     PLOT     #######################
###########################################################
print("\n" + "="*60)
print("Generating plot...")
print("="*60)

fig, ax = plt.subplots(figsize=(8, 5))

# Plot cost vs time
ax.plot(time_FW, cost_FW, linewidth=2, label="Vanilla FW", marker='o', markevery=max(1, len(time_FW)//10))
ax.plot(time_FW_p2, cost_FW_p2, linewidth=2, label="FW p=2", marker='s', markevery=max(1, len(time_FW_p2)//10))
ax.plot(time_FW_trunc, cost_FW_trunc, linewidth=2, label="FW Truncated", marker='^', markevery=max(1, len(time_FW_trunc)//10))

# Set log scale
ax.set_yscale('log')

ax.set_xlabel("Time (seconds)", fontsize=12)
ax.set_ylabel("UOT cost", fontsize=12)
ax.set_title(f"Cost vs Time Comparison (n={n}, R={R}, p={p})", fontsize=13)

ax.legend()
ax.grid(True, alpha=0.3, which='both', linestyle='--')
plt.tight_layout()

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the subfolder path
plot_dir = os.path.join(script_dir, 'Plots', '2dim')

# Ensure directory exists
os.makedirs(plot_dir, exist_ok=True)

# Save file 
plot_path = os.path.join(plot_dir, 
                         f'cost_vs_time_trunc(n{n}_R{R}_p{p}_delta{delta}_eps{eps}_iter{max_iter}).png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

plt.show()

print(f"\nPlot saved to: {plot_path}")
print("\n" + "="*60)
print(f"Summary:")
print(f"  Vanilla FW:    {len(cost_FW)} iterations, final cost = {cost_FW[-1]:.4f}")
print(f"  FW p=2:        {len(cost_FW_p2)} iterations, final cost = {cost_FW_p2[-1]:.4f}")
print(f"  FW Truncated:  {len(cost_FW_trunc)} iterations, final cost = {cost_FW_trunc[-1]:.4f}")
print("="*60)
plt.close()
