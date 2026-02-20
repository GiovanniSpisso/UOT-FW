import numpy as np
import time

from fastuot.uot1d import solve_uot, invariant_dual_loss
from FW_Sejourne import primal_uot_value_from_atoms
from FW_1dim_trunc import PW_FW_dim1_trunc, truncated_cost, UOT_cost_upper

# Parameters
n = 500
p = 1
rho1 = 1.0  # must be set to 1 to obtain same results as FW truncated
R = 7  # Truncation radius
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
delta = 0.001
eps = 0.001

start_time_trunc = time.time()
xk, (grad_xk_x, grad_xk_s), x_marg, y_marg, s_i, s_j = PW_FW_dim1_trunc(
    a, b, M, p, c_trunc, R,
    max_iter=max_iter_fw, delta=delta, eps=eps
)
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
