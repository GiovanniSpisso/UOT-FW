import numpy as np
import time

from fastuot.uot1d import solve_uot, invariant_dual_loss
from FW_Sejourne import primal_uot_value_from_atoms
from FW_truncated import x_init_trunc, grad_trunc, LMO_x, LMO_s, gap_calc_trunc, \
    compute_gamma_max, step_calc as step_calc_trunc, update_grad_trunc, truncated_cost, UOT_cost_upper

# Parameters
n = 500
p = 1
rho1 = 1.0  # must be set to 1 to obtain same results as FW truncated
R = 5  # Truncation radius
max_iter = 1000000

# Generate data
np.random.seed(0)
a = np.random.randint(1, 1001, size=n).astype(float)
b = np.random.randint(1, 1001, size=n).astype(float)
x, y = np.arange(n).astype(float), np.arange(n).astype(float)

###########################################################
#############  SOLVE_UOT (Sejourne)  #####################
###########################################################
print("\n" + "="*60)
print("Running solve_uot (Sinkhorn-based UOT)")
print("="*60)

start_time_sejourne = time.time()
I, J, P, f, g, cost = solve_uot(
    a, b, x, y, p, rho1, niter=max_iter, tol=1e-5,
    greed_init=False, line_search='default', stable_lse=True
)
end_time_sejourne = time.time()

# Compute final costs
primal_cost_sejourne = primal_uot_value_from_atoms(I, J, P, x, y, a, b, p, rho1)
dual_cost_sejourne = invariant_dual_loss(f, g, a, b, rho1)

print(f"Primal Cost: {primal_cost_sejourne:.6f}")
print(f"Dual Cost: {dual_cost_sejourne:.6f}")
print(f"Primal-Dual Gap: {primal_cost_sejourne - dual_cost_sejourne:.6f}")
print(f"Total time: {end_time_sejourne - start_time_sejourne:.4f}s")


###########################################################
##############  Truncated FW  #############################
###########################################################
print("\n" + "="*60)
print("Running Truncated FW")
print("="*60)

n_points = len(a)

# Create truncated cost matrix (vector representation)
c_trunc = np.concatenate([
    np.full(n_points - abs(k), abs(k))
    for k in range(-R + 1, R)
])

# Truncated FW setup
M = 2 * (np.sum(a) + np.sum(b))
max_iter_fw = 10000
delta = 0.01
eps = 0.001

# Initial transportation plan, marginals, cost and gradient
xk, x_marg, y_marg, mask1, mask2 = x_init_trunc(a, b, n_points, c_trunc, p)
s_i, s_j = np.zeros(n_points), np.zeros(n_points)
grad_xk_x, grad_xk_s = grad_trunc(x_marg, y_marg, mask1, mask2, c_trunc, p, n_points, R)

start_time_trunc = time.time()

for k in range(max_iter_fw - 1):
    # LMO calls
    vk_x = LMO_x(xk, grad_xk_x, M, eps)
    vk_s = LMO_s(s_i, s_j, grad_xk_s, M, eps, mask1, mask2)

    # Gap calculation
    gap = gap_calc_trunc(xk, grad_xk_x, vk_x, M, s_i, s_j, grad_xk_s, vk_s)

    if gap <= delta or (vk_x == (-1, -1) and vk_s == (-1, -1, -1, -1)):
        print(f"Truncated FW converged after {k+1} iterations")
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

end_time_trunc = time.time()

# Compute final costs
cost_trunc_final = truncated_cost(xk, x_marg, y_marg, c_trunc, a, b, p, s_i, s_j, R)
cost_full_final = UOT_cost_upper(cost_trunc_final, n_points, s_i, R)

print(f"Truncated Cost: {cost_trunc_final:.6f}")
print(f"Full Cost (upper bound): {cost_full_final:.6f}")
print(f"Total time: {end_time_trunc - start_time_trunc:.4f}s")


###########################################################
################  COMPARISON  ############################
###########################################################
print("\n" + "="*60)
print("FINAL COSTS COMPARISON")
print("="*60)
print(f"Problem size: n = {n}")
print(f"Parameters: p = {p}, rho1 = {rho1}, R = {R}")
print(f"Max iterations: {max_iter}")
print("")
print(f"{'Method':<25} {'Type':<20} {'Cost':<15}")
print("-" * 60)
print(f"{'solve_uot (Sejourne)':<25} {'Primal':<20} {primal_cost_sejourne:<15.6f}")
print(f"{'solve_uot (Sejourne)':<25} {'Dual':<20} {dual_cost_sejourne:<15.6f}")
print(f"{'solve_uot (Sejourne)':<25} {'Primal-Dual Gap':<20} {primal_cost_sejourne - dual_cost_sejourne:<15.6f}")
print("-" * 60)
print(f"{'Truncated FW':<25} {'Truncated':<20} {cost_trunc_final:<15.6f}")
print(f"{'Truncated FW':<25} {'Full (upper bound)':<20} {cost_full_final:<15.6f}")
print("-" * 60)
print(f"\nDifference (Sejourne Primal - Truncated Full): {abs(primal_cost_sejourne - cost_full_final):.6f}")
print(f"Difference (Sejourne Dual - Truncated Full): {abs(dual_cost_sejourne - cost_full_final):.6f}")
print("="*60)
