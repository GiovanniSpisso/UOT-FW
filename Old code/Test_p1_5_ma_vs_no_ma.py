import time
import numpy as np

from FW_1dim_p1_5_np_ma import PW_FW_dim1_p1_5 as PW_masked, cost_p1_5 as cost_masked
from FW_1dim_p1_5_0 import PW_FW_dim1_p1_5 as PW_nomask, cost_p1_5 as cost_nomask

np.set_printoptions(precision=6, suppress=True)


def to_dense(x):
    return x.filled(0.0) if np.ma.isMaskedArray(x) else np.asarray(x, dtype=float)


def run_solver(name, solver, cost_fn, mu, nu, M, max_iter, delta, eps):
    t0 = time.perf_counter()
    xk, grad, x_marg, y_marg = solver(mu.copy(), nu.copy(), M, max_iter=max_iter, delta=delta, eps=eps)
    elapsed = time.perf_counter() - t0

    xk_d = to_dense(xk)
    x_marg_d = to_dense(x_marg)
    y_marg_d = to_dense(y_marg)

    # cost with dense arrays (for consistent comparison)
    c = cost_fn(xk_d, x_marg_d, y_marg_d, mu, nu)

    return {
        "name": name,
        "time": elapsed,
        "plan": xk_d,
        "x_marg": x_marg_d,
        "y_marg": y_marg_d,
        "cost": float(c),
    }


def diff_stats(a, b):
    d = np.asarray(a) - np.asarray(b)
    return {
        "l1": float(np.sum(np.abs(d))),
        "l2": float(np.linalg.norm(d)),
        "linf": float(np.max(np.abs(d))),
    }


def main():
    # Parameters
    seed = 0
    n = 5000
    max_iter = 10000
    delta = 1e-3
    eps = 1e-3
    repeats = 3
    zero_fraction = 0.95  # set 0.0 if you do not want zeros in measures

    rng = np.random.default_rng(seed)
    mu = rng.integers(0, 100, size=n).astype(float)
    nu = rng.integers(0, 100, size=n).astype(float)

    if zero_fraction > 0:
        z_mu = rng.choice(n, size=int(zero_fraction * n), replace=False)
        z_nu = rng.choice(n, size=int(zero_fraction * n), replace=False)
        mu[z_mu] = 0.0
        nu[z_nu] = 0.0

    M = n * (np.sum(mu) + np.sum(nu))

    masked_runs = []
    nomask_runs = []

    print("\nRunning benchmarks...\n")
    for r in range(repeats):
        print(f"Repeat {r+1}/{repeats}")

        out_masked = run_solver(
            "FW_1dim_p1_5_np.ma",
            PW_masked,
            cost_masked,
            mu,
            nu,
            M,
            max_iter,
            delta,
            eps,
        )
        out_nomask = run_solver(
            "FW_1dim_p1_5_0",
            PW_nomask,
            cost_nomask,
            mu,
            nu,
            M,
            max_iter,
            delta,
            eps,
        )

        masked_runs.append(out_masked)
        nomask_runs.append(out_nomask)

        print(f"  np.ma   : {out_masked['time']:.4f} s | cost={out_masked['cost']:.6f}")
        print(f"  p1_5_0  : {out_nomask['time']:.4f} s | cost={out_nomask['cost']:.6f}")

    # Use last run for plan/marginal comparison
    m = masked_runs[-1]
    u = nomask_runs[-1]

    plan_diff = diff_stats(m["plan"], u["plan"])
    xm_diff = diff_stats(m["x_marg"], u["x_marg"])
    ym_diff = diff_stats(m["y_marg"], u["y_marg"])

    t_masked = np.array([x["time"] for x in masked_runs])
    t_nomask = np.array([x["time"] for x in nomask_runs])

    print("\n" + "=" * 64)
    print("TIME COMPARISON")
    print("=" * 64)
    print(f"np.ma  avg: {t_masked.mean():.4f} s  (std {t_masked.std(ddof=0):.4f})")
    print(f"p1_5_0 avg: {t_nomask.mean():.4f} s  (std {t_nomask.std(ddof=0):.4f})")
    if t_nomask.mean() > 0:
        print(f"speedup (np.ma / p1_5_0): {t_masked.mean() / t_nomask.mean():.4f}x")

    print("\n" + "=" * 64)
    print("SOLUTION COMPARISON (LAST REPEAT)")
    print("=" * 64)
    print(f"cost np.ma : {m['cost']:.10f}")
    print(f"cost p1_5_0: {u['cost']:.10f}")
    print(f"|Δcost|    : {abs(m['cost'] - u['cost']):.10e}")

    print("\nPlan diff (3n vector):")
    print(f"  L1   = {plan_diff['l1']:.10e}")
    print(f"  L2   = {plan_diff['l2']:.10e}")
    print(f"  Linf = {plan_diff['linf']:.10e}")

    print("\nx_marg diff:")
    print(f"  L1   = {xm_diff['l1']:.10e}")
    print(f"  L2   = {xm_diff['l2']:.10e}")
    print(f"  Linf = {xm_diff['linf']:.10e}")

    print("\ny_marg diff:")
    print(f"  L1   = {ym_diff['l1']:.10e}")
    print(f"  L2   = {ym_diff['l2']:.10e}")
    print(f"  Linf = {ym_diff['linf']:.10e}")

    print("\nAllclose checks:")
    print(f"  plan   : {np.allclose(m['plan'], u['plan'], rtol=1e-7, atol=1e-9)}")
    print(f"  x_marg : {np.allclose(m['x_marg'], u['x_marg'], rtol=1e-7, atol=1e-9)}")
    print(f"  y_marg : {np.allclose(m['y_marg'], u['y_marg'], rtol=1e-7, atol=1e-9)}")


if __name__ == "__main__":
    main()