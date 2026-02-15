import time
import argparse
import numpy as np
import importlib

# Import both implementations
import FW_1dim_p2 as class_mod
import FW_1dim_p2_array as array_mod


def make_problem(n, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.integers(1, 2001, size=n)
    nu = rng.integers(1, 2001, size=n)
    # simple cost: squared distance
    idx = np.arange(n)
    c = (idx[:, None] - idx[None, :])**2
    M = n * (np.sum(mu) + np.sum(nu))
    return mu, nu, c, M


def run_once_class(n, max_iter, delta=0.01, eps=0.001):
    """Run the class-based implementation (FW_1dim_p2)"""
    mu, nu, c, M = make_problem(n)
    t0 = time.perf_counter()
    xk, grad_xk, x_marg, y_marg = class_mod.PW_FW_dim1_p2(mu, nu, c, M, max_iter=max_iter, delta=delta, eps=eps)
    t1 = time.perf_counter()
    cost = class_mod.cost_p2(xk.to_dense(), x_marg, y_marg, c, mu, nu)
    return (t1 - t0), cost


def run_once_array(n, max_iter, delta=0.01, eps=0.001):
    """Run the array-based implementation (FW_1dim_p2_array)"""
    mu, nu, c, M = make_problem(n)
    t0 = time.perf_counter()
    xk, grad_xk, x_marg, y_marg = array_mod.PW_FW_dim1_p2(mu, nu, M, max_iter=max_iter, delta=delta, eps=eps)
    t1 = time.perf_counter()
    cost = array_mod.cost_p2(xk, x_marg, y_marg, mu, nu)
    return (t1 - t0), cost


def main():
    parser = argparse.ArgumentParser(description="Compare FW p=2 implementations: class vs array-packed tridiagonal")
    parser.add_argument("--n", type=int, default=1000, help="problem size (n)")
    parser.add_argument("--max-iter", type=int, default=1000, help="max iterations")
    parser.add_argument("--profile", action="store_true", help="run cProfile on each implementation")
    args = parser.parse_args()

    n = args.n
    max_iter = args.max_iter

    print(f"Comparing implementations for n={n}, max_iter={max_iter}\n")

    # Warmup imports
    importlib.reload(class_mod)
    importlib.reload(array_mod)

    # Run single tests for each implementation
    t_class, cost_class = run_once_class(n, max_iter)
    t_array, cost_array = run_once_array(n, max_iter)

    print(f"Class:  time={t_class:.4f}s cost={cost_class:.6f}")
    print(f"Array:  time={t_array:.4f}s cost={cost_array:.6f}")

    speedup = t_class / t_array if t_array > 0 else float('inf')
    cost_diff = float(np.abs(cost_class - cost_array))

    print("\nSummary:")
    print(f"Speedup (class / array): {speedup:.3f}x")
    print(f"Cost difference: {cost_diff:.6f}")

    if args.profile:
        print("\nRunning cProfile for class implementation...")
        import cProfile, pstats
        pr = cProfile.Profile()
        pr.enable()
        run_once_class(n, max_iter)
        pr.disable()
        ps = pstats.Stats(pr).sort_stats("cumtime")
        ps.print_stats(20)

        print("\nRunning cProfile for array implementation...")
        pr = cProfile.Profile()
        pr.enable()
        run_once_array(n, max_iter)
        pr.disable()
        ps = pstats.Stats(pr).sort_stats("cumtime")
        ps.print_stats(20)


if __name__ == "__main__":
    main()
