import numpy as np
import matplotlib.pyplot as plt
import os
import time

from FW_Sejourne import solve_uot_with_cost_tracking, primal_uot_value_from_atoms
from FW_1dim_trunc import x_init_trunc, grad_trunc, LMO_x, LMO_s, gap_calc_trunc, \
    compute_gamma_max, step_calc as step_calc_trunc, update_grad_trunc, truncated_cost, UOT_cost_upper

# Parameters
n = 500
p = 1
max_iter = 10000
rho1 = 1  # must be set to 1 to obtain same results as FW truncated
R = 5  # Truncation radius

# Generate data
np.random.seed(0)
a = np.random.randint(1, 1001, size=n).astype(float)
b = np.random.randint(1, 1001, size=n).astype(float)
x, y = np.arange(n).astype(float), np.arange(n).astype(float)
x_marg_in = np.sqrt(a * b) # greedy potential initialization

###########################################################
#############  SOLVE_UOT with cost tracking  #############
###########################################################
print("\n" + "="*60)
print("Running solve_uot (Sinkhorn-based UOT)")
print("="*60)

start_time_sejourne = time.time()
I, J, P, f, g, cost, costs_sejourne, primal_gaps_sejourne, dual_gaps_sejourne = solve_uot_with_cost_tracking(
    a, b, x, y, p, rho1, niter=max_iter, tol=1e-5,
    greed_init=True, line_search='default', stable_lse=True
)
end_time_sejourne = time.time()

# Calculate cumulative times
times_sejourne = np.linspace(0, end_time_sejourne - start_time_sejourne, len(costs_sejourne))

print(f"Final Primal Cost: {costs_sejourne[-1]:.6f}")
print(f"Number of iterations: {len(costs_sejourne)}")
print(f"Total time: {end_time_sejourne - start_time_sejourne:.4f}s")


###########################################################
##############  Truncated FW comparison  #################
###########################################################
print("\n" + "="*60)
print("Running Truncated FW for comparison")
print("="*60)

n_points = len(a)

# Create truncated cost matrix (vector representation)
c_trunc = np.concatenate([
    np.full(n_points - abs(k), abs(k))
    for k in range(-R + 1, R)
])

# Truncated FW setup
M = 2 * (np.sum(a) + np.sum(b))
max_iter_fw = 1000
delta = 0.01
eps = 0.001


# Start the algorithm from 0 plan
xk = np.zeros_like(c_trunc, dtype=float)
x_marg = np.zeros(n_points)
y_marg = np.zeros(n_points)
mask1 = (a != 0)
mask2 = (b != 0)


# Initial transportation plan, marginals, cost and gradient
xk, x_marg, y_marg, mask1, mask2 = x_init_trunc(a, b, n_points, c_trunc, p)
s_i, s_j = np.zeros(n_points), np.zeros(n_points)
grad_xk_x, grad_xk_s = grad_trunc(x_marg, y_marg, mask1, mask2, c_trunc, p, n_points, R)

costs_trunc = [truncated_cost(xk, x_marg, y_marg, c_trunc, a, b, p, s_i, s_j, R)]
cost_full_estimates = [UOT_cost_upper(costs_trunc[0], n_points, s_i, R)]
times_trunc = [0.0]

start_time_trunc = time.time()
k = 0
for k in range(max_iter_fw - 1):
    # LMO calls
    vk_x = LMO_x(xk, grad_xk_x, M, eps)
    vk_s = LMO_s(s_i, s_j, grad_xk_s, M, eps, mask1, mask2)

    # Gap calculation
    gap = gap_calc_trunc(xk, grad_xk_x, vk_x, M, s_i, s_j, grad_xk_s, vk_s)

    if gap <= delta or (vk_x == (-1, -1) and vk_s == (-1, -1, -1, -1)):
        print(f"Truncated FW converged after {k} iterations")
        break
    
    # Unpack search directions
    FW_x, AFW_x = vk_x
    FW_si, FW_sj, AFW_si, AFW_sj = vk_s

    # Compute maximum allowed step size
    gamma_max = compute_gamma_max(xk, s_i, s_j, FW_x, AFW_x, FW_si, AFW_si, FW_sj, AFW_sj, M)
    
    # Compute step size using Armijo
    result = step_calc_trunc(x_marg, y_marg, grad_xk_x, grad_xk_s,
                      a, b, vk_x, vk_s, s_i, s_j, c_trunc, p, n_points, R, 
                      theta=gamma_max)
    
    if isinstance(result, tuple):
        gammak, i_FW, j_FW, i_AFW, j_AFW = result
    else:
        gammak = result
        i_FW, j_FW, i_AFW, j_AFW = -1, -1, -1, -1

    # Update x coordinates
    if AFW_x != -1:
        xk[AFW_x] -= gammak
        if i_AFW != -1:
            x_marg[i_AFW] -= gammak / a[i_AFW]
        if j_AFW != -1:
            y_marg[j_AFW] -= gammak / b[j_AFW]
        
        if FW_x != -1:
            xk[FW_x] += gammak
            if i_FW != -1:
                x_marg[i_FW] += gammak / a[i_FW]
            if j_FW != -1:
                y_marg[j_FW] += gammak / b[j_FW]
    elif FW_x != -1:
        xk[FW_x] += gammak
        if i_FW != -1:
            x_marg[i_FW] += gammak / a[i_FW]
        if j_FW != -1:
            y_marg[j_FW] += gammak / b[j_FW]

    # Update s_i, s_j coordinates
    if FW_si != -1 and FW_si < len(s_i):
        s_i[FW_si] += gammak / a[FW_si]
    if FW_sj != -1 and FW_sj < len(s_j):
        s_j[FW_sj] += gammak / b[FW_sj]
    if AFW_si != -1 and AFW_si < len(s_i):
        s_i[AFW_si] -= gammak / a[AFW_si]
    if AFW_sj != -1 and AFW_sj < len(s_j):
        s_j[AFW_sj] -= gammak / b[AFW_sj]

    # Update gradient
    v_coords = (i_FW, j_FW, i_AFW, j_AFW)
    grad_xk_x, grad_xk_s = update_grad_trunc(x_marg, y_marg, s_i, s_j, grad_xk_x, grad_xk_s, 
                                             mask1, mask2, p, n_points, R, v_coords, vk_s)
    
    if k % 100 == 0:
        cost_trunc = truncated_cost(xk, x_marg, y_marg, c_trunc, a, b, p, s_i, s_j, R)
        costs_trunc.append(cost_trunc)
        cost_full_estimate = UOT_cost_upper(cost_trunc, n_points, s_i, R)
        cost_full_estimates.append(cost_full_estimate)
        
        elapsed_time = time.time() - start_time_trunc
        times_trunc.append(elapsed_time)

end_time_trunc = time.time()

# Record final cost if not already recorded (in case convergence happened between cost updates)
if k % 100 != 0:
    cost_trunc = truncated_cost(xk, x_marg, y_marg, c_trunc, a, b, p, s_i, s_j, R)
    costs_trunc.append(cost_trunc)
    cost_full_estimate = UOT_cost_upper(cost_trunc, n_points, s_i, R)
    cost_full_estimates.append(cost_full_estimate)
    elapsed_time = end_time_trunc - start_time_trunc
    times_trunc.append(elapsed_time)

print(f"Truncated FW final cost: {costs_trunc[-1]:.6f}")
print(f"Truncated FW final cost estimate (upper bound): {cost_full_estimates[-1]:.6f}")
print(f"Truncated FW number of iterations: {len(costs_trunc)}")
print(f"Total time: {end_time_trunc - start_time_trunc:.4f}s")


###########################################################
###################  PLOTTING  ###########################
###########################################################
print("\n" + "="*60)
print("Generating comparison plot...")
print("="*60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# LEFT PLOT: Cost vs Time
ax1.plot(times_sejourne, costs_sejourne, linewidth=2.5, 
        label="solve_uot", marker='o', 
        markevery=max(1, len(costs_sejourne)//20), markersize=6)

ax1.plot(times_trunc, cost_full_estimates, linewidth=2.5, 
        label=f"Truncated FW (R={R})", marker='s', 
        markevery=max(1, len(cost_full_estimates)//20), markersize=6)

ax1.set_yscale('log')
y_max = cost_full_estimates[0] * 1000
ax1.set_ylim(top=y_max)

ax1.set_xlabel("Time (seconds)", fontsize=12)
ax1.set_ylabel("Primal Cost (log scale)", fontsize=12)
ax1.set_title(f"Convergence: Cost vs Time", fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, linestyle='--')

# RIGHT PLOT: Gaps vs Time (only for solve_uot)
ax2.plot(times_sejourne, primal_gaps_sejourne, linewidth=2.5, 
        label="Primal Gap", marker='o', 
        markevery=max(1, len(primal_gaps_sejourne)//20), markersize=6)

ax2.plot(times_sejourne, dual_gaps_sejourne, linewidth=2.5, 
        label="Dual Gap", marker='s', 
        markevery=max(1, len(dual_gaps_sejourne)//20), markersize=6)

# Horizontal line for final truncated FW cost
ax2.axhline(y=cost_full_estimates[-1], color='red', linestyle='--', linewidth=2, 
            label=f"Final Truncated FW Cost")

ax2.set_yscale('log')
ax2.set_xlabel("Time (seconds)", fontsize=12)
ax2.set_ylabel("Gap Value (log scale)", fontsize=12)
ax2.set_title(f"Convergence: Gaps vs Time (solve_uot)", fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()

# Save plot
script_dir = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(script_dir, 'Plots', '1dim')
os.makedirs(plot_dir, exist_ok=True)
plot_path = os.path.join(plot_dir, f'Sejourne_comparison(n{n}_iter{max_iter}_R{R}).png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")

plt.show()

###########################################################
################  SUMMARY  ###############################
###########################################################
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Problem size: n = {n}")
print(f"Parameters: p = {p}, rho1 = {rho1}, R = {R}")
print(f"\nsolve_uot (Sinkhorn):")
print(f"  Iterations: {len(costs_sejourne)}")
print(f"  Final cost: {costs_sejourne[-1]:.6f}")
print(f"\nTruncated FW:")
print(f"  Iterations: {len(costs_trunc)}")
print(f"  Final cost: {cost_full_estimates[-1]:.6f}")
print(f"\nCost difference: {abs(costs_sejourne[-1] - cost_full_estimates[-1]):.6f}")
print("="*60)