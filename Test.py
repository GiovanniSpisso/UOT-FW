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


def run_solver(label, solver, cost_fn, build_c_fn, mu, nu, M, p, R, max_iter, delta, eps):
    t0 = time.perf_counter()
    out = solver(mu.copy(), nu.copy(), M, p, R, max_iter=max_iter, delta=delta, eps=eps)
    elapsed = time.perf_counter() - t0

    # FW_1dim_trunc_0 may return None if max_iter is reached without early return
    if out is None:
        return {"name": label, "time_sec": elapsed, "cost": None, "status": "no return (max_iter reached)"}

    xk, grad, x_marg, y_marg, s_i, s_j = out
    c_trunc = build_c_fn(len(mu), R)
    cost_val = cost_fn(xk, x_marg, y_marg, c_trunc, mu, nu, p, s_i, s_j, R)

    return {"name": label, "time_sec": elapsed, "cost": float(cost_val), "status": "ok"}


if __name__ == "__main__":
    # Same setup for both solvers
    np.random.seed(1)
    n = 500
    p = 1
    R = 10
    max_iter = 10000
    delta = 1e-3
    eps = 1e-3

    # Keep masses strictly positive to avoid division-by-zero issues
    mu = np.random.randint(1, 1000, size=n).astype(np.float64)
    nu = np.random.randint(1, 1000, size=n).astype(np.float64)
    M = n * (np.sum(mu) + np.sum(nu))

    results = []
    results.append(run_solver("FW_1dim_trunc_0", pw_old, truncated_cost_old, build_c_old,
                              mu, nu, M, p, R, max_iter, delta, eps))
    results.append(run_solver("FW_1dim_trunc", pw_new, truncated_cost_new, build_c_new,
                              mu, nu, M, p, R, max_iter, delta, eps))

    print("\n" + "=" * 70)
    print("PW_FW_dim1_trunc benchmark: truncated_cost and elapsed time")
    print("=" * 70)
    for r in results:
        if r["cost"] is None:
            print(f"{r['name']:<18} | time = {r['time_sec']:.6f} s | cost = N/A | {r['status']}")
        else:
            print(f"{r['name']:<18} | time = {r['time_sec']:.6f} s | cost = {r['cost']:.10f}")

    if all(r["cost"] is not None for r in results):
        diff = abs(results[0]["cost"] - results[1]["cost"])
        print("-" * 70)
        print(f"Absolute cost difference: {diff:.10f}")