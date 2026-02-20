import time
import os
import numpy as np
import matplotlib.pyplot as plt
from FW_1dim import x_init, grad, LMO, gap_calc, step_calc, \
    update_grad, UOT_cost, apply_step, update_sum_term
from FW_1dim_p2 import x_init_p2, grad_p2, LMO_p2, gap_calc_p2, \
  update_grad_p2, opt_step, apply_step_p2, update_sum_term_p2, cost_p2, vec_i_to_mat_i_p2
from FW_1dim_trunc import PW_FW_dim1_trunc, truncated_cost

# Set seed for replicability
np.random.seed(0)
# Number of support point
n = 2000  
# Define two positive and discrete measures
mu = np.random.randint(0, 1001, size = n)
nu = np.random.randint(0, 1001, size = n)

# Main parameters to set
c = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) # cost function
p = 2                                                     # power of the entropy
M = 2 * (np.sum(mu) + np.sum(nu))                         # upper bound for delimiting the generalized simplex
R = 2                                                     # truncation radius
max_iter = 2000                          
delta = 0.001                                             # tolerance to stop the gap
eps = 0.001                                               # tolerance for calculating the descent direction

###########################################################
###################     VANILLA FW     ####################
###########################################################
print("\n" + "="*60)
print("Computing Cost VS Time for Vanilla FW")
print("="*60)
n_points = len(mu)

# initial transportation plan, marginals and gradient initialization
cost_list = []
time_list = []
start_time = time.time()
no_time = 0

xk, x_marg, y_marg, mask1, mask2 = x_init(mu, nu, p, n_points)
grad_xk = grad(x_marg, y_marg, mask1, mask2, p, c)

# Initialize sum_term for efficient gap calculation
sum_term = np.sum(grad_xk * xk)
k = 0
for k in range(max_iter):
    if k % 100 == 0:
        tot_time = time.time()
        time_list.append(tot_time - no_time - start_time)
        cost_list.append(UOT_cost(xk, x_marg, y_marg, c, mu, nu, p))
        no_time += time.time() - tot_time

    # search direction
    vk = LMO(xk, grad_xk, M, eps)

    # gap calculation
    gap = gap_calc(grad_xk, vk, M, sum_term)

    if (gap <= delta) or (vk == (-1,-1,-1,-1)):
      print("Converged after: ", k, " iterations.")
      break

    # coordinates + rows and columns update
    FW_i, FW_j, AFW_i, AFW_j = vk

    # Collect affected rows and columns
    rows, cols = set([FW_i, AFW_i]) - {-1}, set([FW_j, AFW_j]) - {-1}

    # Remove contributions from affected rows/columns before gradient update
    sum_term = update_sum_term(sum_term, grad_xk, xk, mask1, mask2, rows, cols, sign=-1)
    
    # Apply step update
    xk, x_marg, y_marg = apply_step(xk, x_marg, y_marg, grad_xk, mu, nu, M, vk, c, p)

    # gradient update
    grad_xk = update_grad(x_marg, y_marg, grad_xk, mask1, mask2, c, vk, p)
    
    # Add back contributions from affected rows/columns after gradient update
    sum_term = update_sum_term(sum_term, grad_xk, xk, mask1, mask2, rows, cols, sign=+1)

# Record final cost if not already recorded
if k % 100 != 0:
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

xk, x_marg, y_marg, mask1, mask2 = x_init_p2(mu, nu, n_points)
grad_xk = grad_p2(x_marg, y_marg, mask1, mask2, n_points)

# Initialize sum_term for efficient gap calculation
sum_term = np.sum(grad_xk * xk)

for k in range(max_iter):
    if k % 100 == 0:
        tot_time = time.time()
        time_list.append(tot_time - no_time - start_time)
        cost_list.append(cost_p2(xk, x_marg, y_marg, mu, nu))
        no_time += time.time() - tot_time

    # search direction (returns 3n vector indices)
    vk = LMO_p2(xk, grad_xk, M, eps)

    # gap calculation
    gap = gap_calc_p2(grad_xk, vk, M, sum_term)

    if (gap <= delta) or (vk == (-1, -1)):
      print("Converged after: ", k, " iterations ")
      break

    # Convert 3n indices to matrix coordinates (done once per iteration)
    FW, AFW = vk
    FW_i, FW_j = vec_i_to_mat_i_p2(FW, n_points) if FW != -1 else (-1, -1)
    AFW_i, AFW_j = vec_i_to_mat_i_p2(AFW, n_points) if AFW != -1 else (-1, -1)
    
    # Remove contributions from affected rows/columns before gradient update
    sum_term = update_sum_term_p2(sum_term, grad_xk, xk, vk, n_points, sign=-1)
    
    # Apply step update
    xk, x_marg, y_marg = apply_step_p2(xk, x_marg, y_marg, mu, nu, M, vk, coords=(FW_i, FW_j, AFW_i, AFW_j))

    # gradient update - pass both 3n indices and coordinates
    grad_xk = update_grad_p2(x_marg, y_marg, grad_xk, mask1, mask2, coords=(FW_i, FW_j, AFW_i, AFW_j))
    
    # Add back contributions from affected rows/columns after gradient update
    sum_term = update_sum_term_p2(sum_term, grad_xk, xk, vk, n_points, sign=+1)

# Record final cost if not already recorded
if k % 100 != 0:
    tot_time = time.time()
    time_list.append(tot_time - no_time - start_time)
    cost_list.append(cost_p2(xk, x_marg, y_marg, mu, nu))

cost_FW_p2 = np.array(cost_list)
time_FW_p2 = np.array(time_list)



###########################################################
################     FW Truncated     #####################
###########################################################
print("\n" + "="*60)
print("Computing Cost VS Time for FW Truncated")
print("="*60)

# Create truncated cost matrix
c_trunc = np.zeros(sum(n_points - abs(k) for k in range(-R + 1, R)))
pos = 0
for k in range(-R + 1, R):
    m = n_points - abs(k)
    if k >= 0:
        i = np.arange(m)
        j = i + k
    else:
        j = np.arange(m)
        i = j - k
    c_trunc[pos:pos + m] = abs(k)
    pos += m

# Run truncated FW with time tracking
cost_list = []
time_list = []
start_time = time.time()

# We need to modify the function to track time, so let's manually run it
from FW_1dim_trunc import x_init_trunc, grad_trunc, LMO_x, LMO_s, gap_calc_trunc, \
    compute_gamma_max, step_calc as step_calc_trunc, update_grad_trunc, \
    vector_to_matrix

n_points = len(mu)

# transportation plan, marginals, cost and gradient initialization
xk, x_marg, y_marg, mask1, mask2 = x_init_trunc(mu, nu, n_points, c_trunc, p)
s_i, s_j = np.zeros(n_points), np.zeros(n_points)
grad_xk_x, grad_xk_s = grad_trunc(x_marg, y_marg, mask1, mask2, c_trunc, p, n_points, R)

no_time = 0

for k in range(max_iter):
    if k % 100 == 0:
        tot_time = time.time()
        time_list.append(tot_time - no_time - start_time)
        cost_list.append(truncated_cost(xk, x_marg, y_marg, c_trunc, mu, nu, p, s_i, s_j, R))
        no_time += time.time() - tot_time
    
    # LMO call
    vk_x = LMO_x(xk, grad_xk_x, M, eps)
    vk_s = LMO_s(s_i, s_j, grad_xk_s, M, eps, mask1, mask2)

    # gap calculation
    gap = gap_calc_trunc(xk, grad_xk_x, vk_x, M, s_i, s_j, grad_xk_s, vk_s)

    if (gap <= delta) or (vk_x == (-1, -1) and vk_s == (-1, -1, -1, -1)):
        print("Converged after: ", k, " iterations ")
        break
    
    # update step
    FW_x, AFW_x = vk_x
    FW_si, FW_sj, AFW_si, AFW_sj = vk_s

    # Compute maximum allowed step size respecting all constraints
    gamma_max = compute_gamma_max(xk, s_i, s_j, FW_x, AFW_x, FW_si, AFW_si, FW_sj, AFW_sj, M)
    
    # Compute step size using Armijo with gamma_max as upper bound
    result = step_calc_trunc(x_marg, y_marg, grad_xk_x, grad_xk_s,
                      mu, nu, vk_x, vk_s, s_i, s_j, c_trunc, p, n_points, R, 
                      theta = gamma_max)
    
    if isinstance(result, tuple):
        gammak, i_FW, j_FW, i_AFW, j_AFW = result
    else:
        gammak = result
        i_FW, j_FW, i_AFW, j_AFW = -1, -1, -1, -1

    # Update x coordinates
    if AFW_x != -1:
        xk[AFW_x] -= gammak
        x_marg[i_AFW] -= gammak / mu[i_AFW]
        y_marg[j_AFW] -= gammak / nu[j_AFW]
        
        if FW_x != -1:
            xk[FW_x] += gammak
            x_marg[i_FW] += gammak / mu[i_FW]
            y_marg[j_FW] += gammak / nu[j_FW]
    elif FW_x != -1:
        xk[FW_x] += gammak
        x_marg[i_FW] += gammak / mu[i_FW]
        y_marg[j_FW] += gammak / nu[j_FW]

    # Update s_i, s_j coordinates
    if FW_si != -1:
        s_i[FW_si] += gammak / mu[FW_si]
        s_j[FW_sj] += gammak / nu[FW_sj]
    if AFW_si != -1:
        s_i[AFW_si] -= gammak / mu[AFW_si]
        s_j[AFW_sj] -= gammak / nu[AFW_sj]

    # update gradient
    v_coords = (i_FW, j_FW, i_AFW, j_AFW)
    grad_xk_x, grad_xk_s = update_grad_trunc(x_marg, y_marg, s_i, s_j, grad_xk_x, grad_xk_s, 
                                             mask1, mask2, p, n_points, R, v_coords, vk_s)

# Record final cost if not already recorded
if k % 100 != 0:
    tot_time = time.time()
    time_list.append(tot_time - no_time - start_time)
    cost_list.append(truncated_cost(xk, x_marg, y_marg, c_trunc, mu, nu, p, s_i, s_j, R))

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
plot_dir = os.path.join(script_dir, 'Plots', '1dim')

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
