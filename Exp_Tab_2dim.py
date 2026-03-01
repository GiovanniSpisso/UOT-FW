import time
import numpy as np
import ot

from FW_2dim import PW_FW_dim2, UOT_cost
from FW_2dim_trunc import (
    PW_FW_dim2_trunc,
    UOT_cost_upper_dim2,
    cost_matrix_trunc_dim2,
    truncated_cost_dim2,
)
from FW_2dim_p2 import PW_FW_dim2_p2, cost_dim2_p2

np.set_printoptions(precision=3, suppress=True)

# ──────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────
np.random.seed(0)

n        = 50
R        = 3
p        = 1
max_iter = 10000
delta    = 0.001
eps      = 0.001

mu = np.random.randint(1, 100, size=(n, n))
nu = np.random.randint(1, 100, size=(n, n))
M  = n * (np.sum(mu) + np.sum(nu))

# Full 2D ground cost c(i,j,k,l) = sqrt((i-k)^2 + (j-l)^2)
idx = np.arange(n)
di = (idx[:, None] - idx[None, :]) ** 2
dj = (idx[:, None] - idx[None, :]) ** 2
c2d = np.sqrt(di[:, None, :, None] + dj[None, :, None, :])   # shape (n,n,n,n)
c_trunc, _ = cost_matrix_trunc_dim2(R)

# POT point clouds and flattened masses
xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
X_a = np.column_stack([xs.ravel(), ys.ravel()])
X_b = X_a.copy()
a = mu.ravel()
b = nu.ravel()

# ──────────────────────────────────────────────
# Solver wrappers
# Each returns:
#   cost, time, plan, x_marg, y_marg, extras
# ──────────────────────────────────────────────
def run_FW_2dim():
    t0 = time.time()
    xk, grad, x_marg, y_marg = PW_FW_dim2(
        mu, nu, M, p, c2d, max_iter=max_iter, delta=delta, eps=eps
    )
    elapsed = time.time() - t0
    cost = UOT_cost(xk, x_marg, y_marg, c2d, mu, nu, p)
    return dict(cost=cost, time=elapsed, plan=xk, x_marg=x_marg, y_marg=y_marg, extras={})

def run_FW_2dim_p2():
    t0 = time.time()
    xk, grad, x_marg, y_marg = PW_FW_dim2_p2(
        mu, nu, M, max_iter=max_iter, delta=delta, eps=eps
    )
    elapsed = time.time() - t0
    cost = cost_dim2_p2(xk, x_marg, y_marg, mu, nu)
    return dict(cost=cost, time=elapsed, plan=xk, x_marg=x_marg, y_marg=y_marg, extras={})

def run_FW_2dim_trunc():
    t0 = time.time()
    xk, (gx, gs), x_marg, y_marg, s_i, s_j = PW_FW_dim2_trunc(
        mu, nu, M, p, R, max_iter=max_iter, delta=delta, eps=eps
    )
    elapsed = time.time() - t0
    cost = truncated_cost_dim2(xk, x_marg, y_marg, c_trunc, mu, nu, p, s_i, s_j, R)
    cost_upper = UOT_cost_upper_dim2(cost, n, s_i, R, mu)
    return dict(
        cost=cost,
        time=elapsed,
        plan=xk,
        x_marg=x_marg,
        y_marg=y_marg,
        extras=dict(s_i=s_i, s_j=s_j, cost_upper=cost_upper),
    )

def run_POT():
    t0 = time.time()
    result = ot.solve_sample(X_a, X_b, a, b, metric="euclidean", unbalanced=1)
    elapsed = time.time() - t0

    plan_flat = result.plan                          # shape (n*n, n*n)
    plan_4d   = plan_flat.reshape(n, n, n, n)       # align with c2d, mu, nu

    x_marg = np.sum(plan_4d, axis=(2, 3)) / mu      # shape (n,n)
    y_marg = np.sum(plan_4d, axis=(0, 1)) / nu      # shape (n,n)

    cost = UOT_cost(plan_4d, x_marg, y_marg, c2d, mu, nu, p)
    return dict(
        cost=cost,
        time=elapsed,
        plan=plan_4d,
        x_marg=x_marg,
        y_marg=y_marg,
        extras=dict(pot_value=result.value, plan_flat=plan_flat),
    )

# ──────────────────────────────────────────────
# SELECT WHICH SOLVERS TO RUN
# Comment out any line to skip that solver.
# ──────────────────────────────────────────────
results = {}
results["FW_2dim_p2"]    = run_FW_2dim_p2()
results["FW_2dim"]       = run_FW_2dim()
#results["FW_2dim_trunc"] = run_FW_2dim_trunc()
#results["POT"]           = run_POT()

# ──────────────────────────────────────────────
# PRINT RESULTS
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("COSTS")
print("=" * 60)
if r := results.get("FW_2dim_p2"):
    print(f"FW_2dim_p2 cost:           {r['cost']:.6f}")
if r := results.get("FW_2dim"):
    print(f"FW_2dim cost:              {r['cost']:.6f}")
if r := results.get("FW_2dim_trunc"):
    print(f"FW_2dim_trunc cost:        {r['cost']:.6f}")
    print(f"FW_2dim_trunc upper bound: {r['extras']['cost_upper']:.6f}")
    print(
        f"S_i total: {np.sum(r['extras']['s_i'] * mu):.6f}   "
        f"S_j total: {np.sum(r['extras']['s_j'] * nu):.6f}"
    )
if r := results.get("POT"):
    print(f"POT cost (my formula):     {r['cost']:.6f}")
    print(f"POT cost (library value):  {r['extras']['pot_value']:.6f}")

print("\n" + "=" * 60)
print("TIMES")
print("=" * 60)
for name, r in results.items():
    print(f"{name:<20} {r['time']:.4f} s")