import time
import os
import numpy as np
import matplotlib.pyplot as plt
from FW_1dim import x_init, UOT_grad, direction, gap_calc, step_calc, \
    UOT_grad_update, UOT_cost
from FW_1dim_p2 import x_init_p2, UOT_grad_p2, direction_class, gap_calc_class, \
  UOT_grad_update_p2, opt_step
import FW_1dim_p2_array as array_mod

# Set seed for replicability
np.random.seed(1)
# Number of support point
n = 2000  
# Define two positive and discrete measures
mu = np.random.randint(0, 101, size=n)
nu = np.random.randint(0, 201, size=n)

# Main parameters to set
c = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))  # cost function
p = 2                                                       # power of the entropy
M = n * (np.sum(mu) + np.sum(nu))                          # upper bound for generalized simplex
max_iter = 10000                          
delta = 0.001                                              # tolerance to stop the gap
eps = 0.001                                                # tolerance for descent direction

###########################################################
###################     VANILLA FW     ####################
###########################################################
print("\nComputing Cost VS Time for Vanilla FW")

# initial transportation plan, marginals and gradient initialization
cost_list = []
time_list = []
start_time = time.time()
no_time = 0

xk, x_marg, y_marg, mask1, mask2 = x_init(mu, nu, p, n)
grad_xk = UOT_grad(x_marg, y_marg, mask1, mask2, p, c)

# Initialize sum_term for efficient gap calculation
sum_term = np.sum(grad_xk * xk)

for k in range(max_iter):
    tot_time = time.time()
    time_list.append(tot_time - no_time - start_time)
    cost_list.append(UOT_cost(xk, x_marg, y_marg, c, mu, nu, p))
    no_time += time.time() - tot_time

    # search direction
    vk = direction(xk, grad_xk, M, eps)

    # gap calculation
    gap = gap_calc(grad_xk, vk, M, sum_term)

    if (gap <= delta) or (vk == (-1,-1,-1,-1)):
      print("Converged after: ", k, " iterations.")
      break

    # coordinates + rows and columns update
    FW_i, FW_j, AFW_i, AFW_j = vk

    # Collect affected rows and columns
    rows, cols = set([FW_i, AFW_i]) - {-1}, set([FW_j, AFW_j]) - {-1}

    # Remove contributions before gradient update
    for i in rows:
      for j in range(n):
        sum_term -= grad_xk[i, j] * xk[i, j]

    for j in cols:
      for i in range(n):
        sum_term -= grad_xk[i, j] * xk[i, j]

    # Add back intersection (entries subtracted twice)
    for i in rows:
      for j in cols:
        sum_term += grad_xk[i, j] * xk[i, j]

    if AFW_i != -1:
      gamma0 = xk[AFW_i, AFW_j] - 1e-10
      # stepsize
      gammak = step_calc(x_marg, y_marg, grad_xk, mu, nu, vk, c, p, theta = gamma0)
      xk[AFW_i, AFW_j] -= gammak
      x_marg[AFW_i] -= gammak / mu[AFW_i]
      y_marg[AFW_j] -= gammak / nu[AFW_j]
      if FW_i != -1:
        xk[FW_i, FW_j] += gammak
        x_marg[FW_i] += gammak / mu[FW_i]
        y_marg[FW_j] += gammak / nu[FW_j]
    else:
      gamma0 = M - np.sum(xk) + xk[FW_i, FW_j]
      # stepsize
      gammak = step_calc(x_marg, y_marg, grad_xk, mu, nu, vk, c, p, theta = gamma0)
      xk[FW_i, FW_j] += gammak
      x_marg[FW_i] += gammak / mu[FW_i]
      y_marg[FW_j] += gammak / nu[FW_j]

    # gradient update
    grad_xk = UOT_grad_update(x_marg, y_marg, grad_xk, mask1, mask2, c, vk, p)

    # Add back contributions after gradient update
    for i in rows:
      sum_term += grad_xk[i, j] * xk[i, j]

    for j in cols:
      sum_term += grad_xk[i, j] * xk[i, j]

    # Remove intersection again (to correct for double addition)
    for i in rows:
      for j in cols:
        sum_term -= grad_xk[i, j] * xk[i, j]

cost_FW = np.array(cost_list)
time_FW = np.array(time_list)


###########################################################
###################     FW for p=2 (CLASS)    #############
###########################################################
print("\nComputing Cost VS Time for FW p=2 (Class)")

# transportation plan, marginals and gradient initialization
cost_list = []
time_list = []
start_time = time.time()
no_time = 0

xk, x_marg, y_marg, mask1, mask2 = x_init_p2(mu, nu)
grad_xk = UOT_grad_p2(x_marg, y_marg, mask1, mask2, c, n)

for k in range(max_iter):
    tot_time = time.time()
    time_list.append(tot_time - no_time - start_time)
    cost_list.append(UOT_cost(xk.to_dense(), x_marg, y_marg, c, mu, nu, p))
    no_time += time.time() - tot_time

    # search direction
    vk = direction_class(xk, grad_xk, M, eps)

    # gap calculation
    gap = gap_calc_class(xk, grad_xk, vk, M)

    if (gap <= delta) or (vk == (-1,-1,-1,-1)):
      print("Converged after: ", k, " iterations.")
      break

    # coordinates + rows and columns update
    xFW, yFW, xAFW, yAFW = vk
    if xAFW != -1:
      gamma0 = xk.get(xAFW, yAFW) - 1e-10
      gammak = min(opt_step(x_marg, y_marg, c, mu, nu, vk), gamma0)
      xk.update(xAFW, yAFW, -gammak)
      x_marg[xAFW] -= gammak / mu[xAFW]
      y_marg[yAFW] -= gammak / nu[yAFW]
      if xFW != -1:
        xk.update(xFW, yFW, gammak)
        x_marg[xFW] += gammak / mu[xFW]
        y_marg[yFW] += gammak / nu[yFW]
    else:
      gamma0 = M - np.sum(x_marg * mu) + xk.get(xFW, yFW)
      gammak = min(opt_step(x_marg, y_marg, c, mu, nu, vk), gamma0)
      xk.update(xFW, yFW, gammak)
      x_marg[xFW] += gammak / mu[xFW]
      y_marg[yFW] += gammak / nu[yFW]

    # gradient update
    grad_xk = UOT_grad_update_p2(x_marg, y_marg, grad_xk, mask1, mask2, c, vk)

cost_FW_p2_class = np.array(cost_list)
time_FW_p2_class = np.array(time_list)


###########################################################
###################   FW for p=2 (ARRAY)    ##############
###########################################################
print("\nComputing Cost VS Time for FW p=2 (Array)")

# transportation plan, marginals and gradient initialization
cost_list = []
time_list = []
start_time = time.time()
no_time = 0

xk, x_marg, y_marg, mask1, mask2 = array_mod.x_init_p2(mu, nu, n)
grad_xk = array_mod.grad_p2(x_marg, y_marg, mask1, mask2, n)

# Initialize sum_term for efficient gap calculation
sum_term = np.sum(grad_xk * xk)

for k in range(max_iter):
    tot_time = time.time()
    time_list.append(tot_time - no_time - start_time)
    cost_list.append(array_mod.cost_p2(xk, x_marg, y_marg, mu, nu))
    no_time += time.time() - tot_time

    # search direction
    vk = array_mod.LMO(xk, grad_xk, M, eps)

    # gap calculation
    gap = array_mod.gap_calc(grad_xk, vk, M, sum_term)

    if (gap <= delta) or (vk == (-1, -1)):
      print("Converged after: ", k, " iterations.")
      break

    # Convert 3n indices to matrix coordinates (done once per iteration)
    FW, AFW = vk
    FW_i, FW_j = (-1, -1)
    AFW_i, AFW_j = (-1, -1)
    
    if FW != -1:
      result = array_mod.vec_i_to_mat_i_p2(FW, n)
      if result is not None:
        FW_i, FW_j = result
    
    if AFW != -1:
      result = array_mod.vec_i_to_mat_i_p2(AFW, n)
      if result is not None:
        AFW_i, AFW_j = result
    
    # rows and columns update
    rows, cols = set([FW_i, AFW_i]) - {-1}, set([FW_j, AFW_j]) - {-1}
    rows, cols = list(rows), list(cols)
    
    # Remove contributions from affected rows/columns before gradient update
    sum_term = array_mod.update_sum_term_p2(sum_term, grad_xk, xk, mask1, mask2, rows, cols, n, sign=-1)
    
    # Apply step update
    xk, x_marg, y_marg = array_mod.apply_step_p2(xk, x_marg, y_marg, mu, nu, M, vk,
                           coords=(FW_i, FW_j, AFW_i, AFW_j))

    # gradient update - pass both 3n indices and coordinates
    grad_xk = array_mod.update_grad_p2(x_marg, y_marg, grad_xk, mask1, mask2,
                       coords=(FW_i, FW_j, AFW_i, AFW_j))
    
    # Add back contributions from affected rows/columns after gradient update
    sum_term = array_mod.update_sum_term_p2(sum_term, grad_xk, xk, mask1, mask2, rows, cols, n, sign=+1)

cost_FW_p2_array = np.array(cost_list)
time_FW_p2_array = np.array(time_list)


# PLOT
plt.figure(figsize=(8, 6))

# Plot cost vs time
plt.plot(time_FW, cost_FW, linewidth=2, label="FW (Vanilla)", marker='o', markersize=4, markevery=max(1, len(time_FW)//20))
plt.plot(time_FW_p2_class, cost_FW_p2_class, linewidth=2, label="FW p=2 (Class)", marker='s', markersize=4, markevery=max(1, len(time_FW_p2_class)//20))
plt.plot(time_FW_p2_array, cost_FW_p2_array, linewidth=2, label="FW p=2 (Array)", marker='^', markersize=4, markevery=max(1, len(time_FW_p2_array)//20))

plt.yscale('log')

plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("UOT cost", fontsize=12)
plt.title("Cost vs Time Comparison", fontsize=13)

plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the subfolder path
plot_dir = os.path.join(script_dir, 'Plots', '1dim')

# Save file 
plot_path = os.path.join(plot_dir, 
                         f'cost_vs_time_comparison_p2(n{n}_p{p}_delta{delta}_eps{eps}_iter{max_iter}).png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"\nPlot saved to: {plot_path}")
plt.close()

# Print summary
print("\nSummary:")
print(f"Vanilla FW converged in {len(time_FW)} iterations, time: {time_FW[-1]:.4f}s, final cost: {cost_FW[-1]:.6f}")
print(f"FW p=2 (Class) converged in {len(time_FW_p2_class)} iterations, time: {time_FW_p2_class[-1]:.4f}s, final cost: {cost_FW_p2_class[-1]:.6f}")
print(f"FW p=2 (Array) converged in {len(time_FW_p2_array)} iterations, time: {time_FW_p2_array[-1]:.4f}s, final cost: {cost_FW_p2_array[-1]:.6f}")
