import time
import os
import numpy as np
import matplotlib.pyplot as plt
from FW_1dim import x_init, grad, LMO, gap_calc, step_calc, \
  update_grad, UOT_cost, update_sum_term, apply_step
from FW_1dim_p1_5 import x_init_p1_5, grad_p1_5, LMO_p1_5, gap_calc_p1_5, \
  update_grad_p1_5, cost_p1_5, update_sum_term_p1_5, apply_step_p1_5, vec_i_to_mat_i_p1_5

# Set seed for replicability
np.random.seed(0)
# Number of support point
n = 2000
# Define two positive and discrete measures
mu = np.random.randint(1, 2001, size = n)
nu = np.random.randint(1, 2001, size = n)

# Main parameters to set
c = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) # cost function
p = 1.5                                                   # power of the entropy
M = 2 * (np.sum(mu) + np.sum(nu))                         # upper bound for delimiting the generalized simplex
max_iter = 1000
delta = 0.001                                             # tolerance to stop the gap
eps = 0.001                                               # tolerance for calculating the descent direction

###########################################################
###################     VANILLA FW     ####################
###########################################################
print("\nComputing Cost VS Time for Vanilla FW")
n = np.shape(mu)[0]

# initial transportation plan, marginals and gradient initialization
cost_list = []
time_list = []
start_time = time.time()
no_time = 0

xk, x_marg, y_marg, mask1, mask2 = x_init(mu, nu, p, n)
grad_xk = grad(x_marg, y_marg, mask1, mask2, p, c)

# Initialize sum_term for efficient gap calculation
sum_term = np.sum(grad_xk * xk)

for k in range(max_iter):
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

    # rows and columns update
    rows, cols = set([FW_i, AFW_i]) - {-1}, set([FW_j, AFW_j]) - {-1}
    rows, cols = list(rows), list(cols)

    # Remove contributions from affected rows/columns before gradient update
    sum_term = update_sum_term(sum_term, grad_xk, xk, mask1, mask2, rows, cols, sign=-1)

    # Apply step update
    xk, x_marg, y_marg = apply_step(xk, x_marg, y_marg, grad_xk, mu, nu, M, vk, c, p)

    # gradient update
    grad_xk = update_grad(x_marg, y_marg, grad_xk, mask1, mask2, c, vk, p)

    # Add back contributions from affected rows/columns after gradient update
    sum_term = update_sum_term(sum_term, grad_xk, xk, mask1, mask2, rows, cols, sign=+1)

cost_FW = np.array(cost_list)
time_FW = np.array(time_list)



###########################################################
##################     FW for p=1.5     ###################
###########################################################
print("\nComputing Cost VS Time for FW for p=1.5")

# transportation plan, marginals and gradient initialization
cost_list = []
time_list = []
start_time = time.time()
no_time = 0

xk, x_marg, y_marg, mask1, mask2 = x_init_p1_5(mu, nu, n)
grad_xk = grad_p1_5(x_marg, y_marg, mask1, mask2, n)

# Initialize sum_term for efficient gap calculation
sum_term = np.sum(grad_xk * xk)

for k in range(max_iter):
    tot_time = time.time()
    time_list.append(tot_time - no_time - start_time)
    cost_list.append(cost_p1_5(xk, x_marg, y_marg, mu, nu))
    no_time += time.time() - tot_time

    # search direction
    vk = LMO_p1_5(xk, grad_xk, M, eps)

    # gap calculation
    gap = gap_calc_p1_5(grad_xk, vk, M, sum_term)

    if (gap <= delta) or (vk == (-1, -1)):
      print("Converged after: ", k, " iterations ")
      break

    # Convert 7n indices to matrix coordinates
    FW, AFW = vk
    FW_i, FW_j = (-1, -1)
    AFW_i, AFW_j = (-1, -1)

    if FW != -1:
      result = vec_i_to_mat_i_p1_5(FW, n)
      if result is not None:
        FW_i, FW_j = result

    if AFW != -1:
      result = vec_i_to_mat_i_p1_5(AFW, n)
      if result is not None:
        AFW_i, AFW_j = result

    # Remove contributions from affected entries before gradient update
    sum_term = update_sum_term_p1_5(sum_term, grad_xk, xk, (FW_i, FW_j, AFW_i, AFW_j), n, sign=-1)

    # Apply step update
    xk, x_marg, y_marg = apply_step_p1_5(
        xk, x_marg, y_marg, grad_xk, mu, nu, M, vk,
        coords=(FW_i, FW_j, AFW_i, AFW_j)
    )

    # gradient update
    grad_xk = update_grad_p1_5(x_marg, y_marg, grad_xk, mask1, mask2,
                               coords=(FW_i, FW_j, AFW_i, AFW_j))

    # Add back contributions from affected entries after gradient update
    sum_term = update_sum_term_p1_5(sum_term, grad_xk, xk, (FW_i, FW_j, AFW_i, AFW_j), n, sign=+1)

cost_FW_p1_5 = np.array(cost_list)
time_FW_p1_5 = np.array(time_list)




# PLOT
plt.figure(figsize=(6, 4))

# Plot cost vs time
plt.plot(time_FW, cost_FW, linewidth=2, label="FW (Vanilla)", marker='o',
         markersize=4, markevery=max(1, len(time_FW)//20))
plt.plot(time_FW_p1_5, cost_FW_p1_5, linewidth=2, label="FW p=1.5", marker='s',
         markersize=4, markevery=max(1, len(time_FW_p1_5)//20))

plt.yscale('log')

plt.xlabel("Time", fontsize=12)
plt.ylabel("UOT cost", fontsize=12)
plt.title("Cost vs Time", fontsize=13)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the subfolder path
plot_dir = os.path.join(script_dir, 'Plots', '1dim')

# Save file 
plot_path = os.path.join(plot_dir, 
                         f'cost_vs_time_p1_5(n{n}_delta{delta}_eps{eps}_iter{max_iter}).png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved to: {plot_path}")
plt.close()