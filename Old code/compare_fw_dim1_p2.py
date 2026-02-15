import time
import numpy as np

import FW_1dim_p2 as fw_class
import FW_1dim_p2_array as fw_array


def make_problem(n, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.integers(0, 1000, size=n).astype(float)
    nu = rng.integers(0, 5000, size=n).astype(float)
    idx = np.arange(n)
    c = np.abs(idx[:, None] - idx[None, :])
    M = n * (np.sum(mu) + np.sum(nu))
    return mu, nu, c, M


def main():
    n = 3000
    max_iter = 100000
    delta = 0.001
    eps = 0.001

    mu, nu, c, M = make_problem(n)

    t0 = time.perf_counter()
    xk_class, grad_class, x_marg_class, y_marg_class = fw_class.PW_FW_dim1_p2(
        mu, nu, c, M, max_iter=max_iter, delta=delta, eps=eps
    )
    t1 = time.perf_counter()
    cost_class = fw_class.cost_p2(xk_class.to_dense(), x_marg_class, y_marg_class, c, mu, nu)
    class_time = t1 - t0

    t2 = time.perf_counter()
    xk_array, grad_array, x_marg_array, y_marg_array = fw_array.PW_FW_dim1_p2(
        mu, nu, M, max_iter=max_iter, delta=delta, eps=eps
    )
    t3 = time.perf_counter()
    cost_array = fw_array.cost_p2(xk_array, x_marg_array, y_marg_array, mu, nu)
    array_time = t3 - t2

    xk_array_dense = fw_array.vec_to_mat_p2(xk_array, n)
    plan_diff_norm = np.linalg.norm(xk_class.to_dense() - xk_array_dense)

    print("FW_dim1_p2 (class) cost:", cost_class)
    print("FW_dim1_p2 (class) time (s):", class_time)
    print("--------------------------------")
    print("FW_dim1_p2_array cost:", cost_array)
    print("FW_dim1_p2_array time (s):", array_time)
    print("--------------------------------")
    print("Absolute cost diff:", abs(cost_class - cost_array))
    print("Plan L2 diff (dense):", plan_diff_norm)


if __name__ == "__main__":
    main()
