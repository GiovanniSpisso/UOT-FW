import time
import numpy as np

from FW_1dim_trunc import (
    PW_FW_dim1_trunc as pw_new,
    truncated_cost as truncated_cost_new,
    build_c as build_c_new,
)
from FW_1dim_trunc_0 import (
    PW_FW_dim1_trunc as pw_old,
    truncated_cost as truncated_cost_old,
    build_c as build_c_old,
)

np.set_printoptions(precision=3, suppress=True)

# ──────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────
n = 2000
p = 1
R = 3
max_iter = 15000
delta = 1e-3
eps = 1e-3

np.random.seed(1)
mu = np.random.randint(1, 1000, size=n).astype(np.float64)
nu = np.random.randint(1, 1000, size=n).astype(np.float64)
M = n * (np.sum(mu) + np.sum(nu))


# ──────────────────────────────────────────────
# Solver wrappers
# Each returns: cost, time, status, extras
# ──────────────────────────────────────────────
def run_fw_1dim_trunc():
    t0 = time.perf_counter()
    out = pw_new(mu.copy(), nu.copy(), M, p, R, max_iter=max_iter, delta=delta, eps=eps)
    elapsed = time.perf_counter() - t0

    if out is None:
        return dict(cost=None, time=elapsed, status="no return (max_iter reached)", extras={})

    xk, (gx, gs), x_marg, y_marg, s_i, s_j = out
    c_trunc = build_c_new(n, R)
    cost = truncated_cost_new(xk, x_marg, y_marg, c_trunc, mu, nu, p, s_i, s_j, R)

    return dict(
        cost=float(cost),
        time=elapsed,
        status="ok",
        extras={
            "s_i_sum": float(np.sum(s_i * mu)),
            "s_j_sum": float(np.sum(s_j * nu)),
        },
    )


def run_fw_1dim_trunc_0():
    t0 = time.perf_counter()
    out = pw_old(mu.copy(), nu.copy(), M, p, R, max_iter=max_iter, delta=delta, eps=eps)
    elapsed = time.perf_counter() - t0

    if out is None:
        return dict(cost=None, time=elapsed, status="no return (max_iter reached)", extras={})

    xk, grad, x_marg, y_marg, s_i, s_j = out
    c_trunc = build_c_old(n, R)
    cost = truncated_cost_old(xk, x_marg, y_marg, c_trunc, mu, nu, p, s_i, s_j, R)

    return dict(
        cost=float(cost),
        time=elapsed,
        status="ok",
        extras={
            "s_i_sum": float(np.sum(s_i * mu)),
            "s_j_sum": float(np.sum(s_j * nu)),
        },
    )


# ──────────────────────────────────────────────
# SELECT WHICH SOLVERS TO RUN
# Comment out one line to skip that solver.
# ──────────────────────────────────────────────
results = {}
results["FW_1dim_trunc_0"] = run_fw_1dim_trunc_0()
results["FW_1dim_trunc"] = run_fw_1dim_trunc()

# ──────────────────────────────────────────────
# PRINT RESULTS
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("COSTS")
print("=" * 60)
for name, r in results.items():
    if r["cost"] is None:
        print(f"{name:<20} N/A ({r['status']})")
    else:
        print(f"{name:<20} {r['cost']:.10f}")
        if "s_i_sum" in r["extras"] and "s_j_sum" in r["extras"]:
            print(
                f"{'':<20} S_i total: {r['extras']['s_i_sum']:.6f}   "
                f"S_j total: {r['extras']['s_j_sum']:.6f}"
            )

print("\n" + "=" * 60)
print("TIMES")
print("=" * 60)
for name, r in results.items():
    print(f"{name:<20} {r['time']:.6f} s")