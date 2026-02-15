import time
import os
import numpy as np
import matplotlib.pyplot as plt
from FW_2dim import x_init, grad_init, direction, gap_calc, step_calc, \
    grad_update, UOT_cost
from FW_2dim_p2 import x_init_p2, grad_init_p2, direction_p2, gap_calc_p2, \
    step_calc_p2, grad_update_p2

# Set seed for replicability
np.random.seed(1)
# Number of support point
n = 40  
# Define two positive and discrete measures
mu = np.random.randint(1, 2001, size = (n,n))
nu = np.random.randint(1, 2001, size = (n,n))

# Main parameters to set
p = 2                                                     # power of the entropy
M = 2 * (np.sum(mu) + np.sum(nu))                         # upper bound for delimiting the generalized simplex
max_iter = 100                          
delta = 0.001                                             # tolerance to stop the gap
eps = 0.001                                               # tolerance for calculating the descent direction

###########################################################
###################     VANILLA FW     ####################
###########################################################
print("\nComputing Cost VS Time for Vanilla FW")
n = len(mu)

# initial transportation plan, marginals and gradient initialization
cost_list = []
time_list = []
start_time = time.time()
no_time = 0

xk, x_marg, y_marg, mask1, mask2 = x_init(mu, nu, p, n)
grad_xk = grad_init(x_marg, y_marg, mask1, mask2, p, n)

# Initialize sum_term for efficient gap calculation
sum_term = np.sum(grad_xk * xk)

for k in range(max_iter):
    tot_time = time.time()
    time_list.append(tot_time - no_time - start_time)
    cost_list.append(UOT_cost(xk, x_marg, y_marg, mu, nu, p))
    no_time += time.time() - tot_time
    # search direction vertices
    vk = direction(xk, grad_xk, M, eps)

    # gap calculation
    gap = gap_calc(grad_xk, vk, M, sum_term)

    if (gap <= delta) or (vk == ((-1,-1,-1,-1),(-1,-1,-1,-1))): 
      print("Converged after: ", k, " iterations ")
      break

    # coordinates + rows and columns update
    (x1FW, x2FW, y1FW, y2FW), (x1AFW, x2AFW, y1AFW, y2AFW) = vk
    
    # Collect affected target and source coordinates
    target_coords = set([(y1FW, y2FW), (y1AFW, y2AFW)]) - {(-1, -1)}
    source_coords = set([(x1FW, x2FW), (x1AFW, x2AFW)]) - {(-1, -1)}
    
    # Remove contributions before gradient update
    # For each affected target (y1, y2): remove grad[:,:,y1,y2] * xk[:,:,y1,y2]
    for y1, y2 in target_coords:
      sum_term -= np.sum(grad_xk[:, :, y1, y2] * xk[:, :, y1, y2])
    
    # For each affected source (x1, x2): remove grad[x1,x2,:,:] * xk[x1,x2,:,:]
    for x1, x2 in source_coords:
      sum_term -= np.sum(grad_xk[x1, x2, :, :] * xk[x1, x2, :, :])
    
    # Add back intersection (entries subtracted twice)
    for x1, x2 in source_coords:
      for y1, y2 in target_coords:
        sum_term += grad_xk[x1, x2, y1, y2] * xk[x1, x2, y1, y2]
    
    if x1AFW != -1:
      gammak = step_calc(x_marg, y_marg, grad_xk, mu, nu, vk, p = p, 
                         step = "optimal", theta = xk[x1AFW, x2AFW, y1AFW, y2AFW])

      xk[x1AFW, x2AFW, y1AFW, y2AFW] -= gammak
      x_marg[x1AFW, x2AFW] -= gammak / mu[x1AFW, x2AFW]
      y_marg[y1AFW, y2AFW] -= gammak / nu[y1AFW, y2AFW]
      if x1FW != -1:

        xk[x1FW, x2FW, y1FW, y2FW] += gammak
        x_marg[x1FW, x2FW] += gammak / mu[x1FW, x2FW]
        y_marg[y1FW, y2FW] += gammak / nu[y1FW, y2FW]
    else:
      # stepsize
      gammak = step_calc(x_marg, y_marg, grad_xk, mu, nu, vk,
                         p = p, step = "optimal", theta = M - np.sum(xk) + xk[x1FW, x2FW, y1FW, y2FW])

      xk[x1FW, x2FW, y1FW, y2FW] += gammak
      x_marg[x1FW, x2FW] += gammak / mu[x1FW, x2FW]
      y_marg[y1FW, y2FW] += gammak / nu[y1FW, y2FW]

    # gradient update
    grad_xk = grad_update(x_marg, y_marg, grad_xk, mask1, mask2, vk, p)
    
    # Add back contributions after gradient update
    # For each affected target (y1, y2): add grad[:,:,y1,y2] * xk[:,:,y1,y2]
    for y1, y2 in target_coords:
      sum_term += np.sum(grad_xk[:, :, y1, y2] * xk[:, :, y1, y2])
    
    # For each affected source (x1, x2): add grad[x1,x2,:,:] * xk[x1,x2,:,:]
    for x1, x2 in source_coords:
      sum_term += np.sum(grad_xk[x1, x2, :, :] * xk[x1, x2, :, :])
    
    # Remove intersection again (to correct for double addition)
    for x1, x2 in source_coords:
      for y1, y2 in target_coords:
        sum_term -= grad_xk[x1, x2, y1, y2] * xk[x1, x2, y1, y2]

cost_FW = np.array(cost_list)
time_FW = np.array(time_list)



###########################################################
###################     FW for p=2     ####################
###########################################################
print("\nComputing Cost VS Time for FW for p=2")

# Parameters
max_iter = 500

# transportation plan, marginals and gradient initialization
cost_list = []
time_list = []
start_time = time.time()
no_time = 0

xk, x_marg, y_marg, mask1, mask2 = x_init_p2(mu, nu)
grad_xk = grad_init_p2(x_marg, y_marg, mask1, mask2, n)


for k in range(max_iter):
    tot_time = time.time()
    time_list.append(tot_time - no_time - start_time)
    cost_list.append(UOT_cost(xk.to_dense(), x_marg, y_marg, mu, nu, p))
    no_time += time.time() - tot_time

    # search direction vertices
    vk = direction_p2(xk, grad_xk, M, eps)

    # gap calculation
    gap = gap_calc_p2(xk, grad_xk, vk, M)

    if (gap <= delta) or (vk == ((-1,-1,-1,-1),(-1,-1,-1,-1))): 
      print("Converged after: ", k, " iterations ")

    # coordinates + rows and columns update
    (x1FW, x2FW, y1FW, y2FW), (x1AFW, x2AFW, y1AFW, y2AFW) = vk
    if x1AFW != -1:

      gammak = step_calc_p2(x_marg, y_marg, grad_xk, mu, nu, vk, p = 2, 
                            step = "optimal", theta = xk.get(x1AFW, x2AFW, y1AFW, y2AFW))

      xk.update(x1AFW, x2AFW, y1AFW, y2AFW, -gammak)
      x_marg[x1AFW, x2AFW] -= gammak / mu[x1AFW, x2AFW]
      y_marg[y1AFW, y2AFW] -= gammak / nu[y1AFW, y2AFW]
      if x1FW != -1:

        xk.update(x1FW, x2FW, y1FW, y2FW, gammak)
        x_marg[x1FW, x2FW] += gammak / mu[x1FW, x2FW]
        y_marg[y1FW, y2FW] += gammak / nu[y1FW, y2FW]
    else:
      # stepsize
      gammak = step_calc_p2(x_marg, y_marg, grad_xk, mu, nu, vk,
                         p = 2, step = "optimal", theta = M - xk.sum() + xk.get(x1FW, x2FW, y1FW, y2FW))

      xk.update(x1FW, x2FW, y1FW, y2FW, gammak)
      x_marg[x1FW, x2FW] += gammak / mu[x1FW, x2FW]
      y_marg[y1FW, y2FW] += gammak / nu[y1FW, y2FW]

    # gradient update
    grad_xk = grad_update_p2(x_marg, y_marg, grad_xk, mask1, mask2, vk)


cost_FW_p2 = np.array(cost_list)
time_FW_p2 = np.array(time_list)




# PLOT
plt.figure(figsize=(6, 4))

# Plot cost vs time
plt.plot(time_FW, cost_FW, linewidth=2, label="FW")
plt.plot(time_FW_p2, cost_FW_p2, linewidth=2, label="FW p=2")

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
plot_dir = os.path.join(script_dir, 'Plots', '2dim')

# Save file 
plot_path = os.path.join(plot_dir, 
                         f'cost_vs_time_2d_p{p}(n{n}_delta{delta}_eps{eps}).png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved to: {plot_path}")
plt.close()