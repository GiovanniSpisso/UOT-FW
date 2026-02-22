import argparse
import numpy as np

from FW_2dim import (
    UOT_cost,
    LMO_dim2,
    PW_FW_dim2,
    apply_step_dim2,
    gap_calc_dim2,
    grad_dim2,
    grad_update_dim2,
    step_calc_dim2,
    update_sum_term_dim2,
    x_init_dim2,
)
from FW_2dim_p2 import (
    LMO_dim2_p2,
    apply_step_dim2_p2,
    cost_dim2_p2,
    gap_calc_dim2_p2,
    grad_dim2_p2,
    grad_update_dim2_p2,
    opt_step_dim2_p2,
    to_dense_dim2_p2,
    update_sum_term_dim2_p2,
    x_init_dim2_p2,
)


def build_9stencil_cost_dense(n, large_cost=1e6):
    c = np.full((n, n, n, n), large_cost, dtype=float)

    offsets = [
        (0, 0, 0.0),
        (-1, 0, 1.0),
        (0, -1, 1.0),
        (1, 0, 1.0),
        (0, 1, 1.0),
        (-1, -1, np.sqrt(2.0)),
        (1, -1, np.sqrt(2.0)),
        (-1, 1, np.sqrt(2.0)),
        (1, 1, np.sqrt(2.0)),
    ]

    for i in range(n):
        for j in range(n):
            for di, dj, val in offsets:
                k, l = i + di, j + dj
                if 0 <= k < n and 0 <= l < n:
                    c[i, j, k, l] = val

    return c


def dense_step_size(xk, x_marg, y_marg, grad_xk, mu, nu, M, v, c):
    (x1FW, x2FW, y1FW, y2FW), (x1AFW, x2AFW, y1AFW, y2AFW) = v
    if x1AFW != -1:
        theta = xk[x1AFW, x2AFW, y1AFW, y2AFW]
    else:
        theta = M - np.sum(xk) + xk[x1FW, x2FW, y1FW, y2FW]

    return step_calc_dim2(
        x_marg, y_marg, grad_xk, mu, nu, v, c, p=2, theta=theta
    )


def compact_step_size(xk, x_marg, y_marg, mu, nu, M, comp_FW, full_FW, comp_AFW, full_AFW):
    mat_FW, i_FW, j_FW = comp_FW
    mat_AFW, i_AFW, j_AFW = comp_AFW

    gamma_opt = opt_step_dim2_p2(
        x_marg,
        y_marg,
        mu,
        nu,
        mat_idx=(mat_FW, mat_AFW),
        full=(full_FW, full_AFW),
    )

    if full_AFW[0] != -1:
        gamma0 = xk[mat_AFW, i_AFW, j_AFW] - 1e-10
    else:
        gamma0 = M - np.sum(xk) + xk[mat_FW, i_FW, j_FW] - 1e-10

    return min(gamma_opt, gamma0)


def format_lmo_compact(lmo):
    (comp_FW, full_FW), (comp_AFW, full_AFW) = lmo
    return {
        "compact_FW": comp_FW,
        "full_FW": full_FW,
        "compact_AFW": comp_AFW,
        "full_AFW": full_AFW,
    }


def compare_iteration_by_iteration(mu, nu, M, max_iter=30, delta=1e-2, eps=1e-3):
    n = mu.shape[0]
    c_dense = build_9stencil_cost_dense(n)

    # Dense (FW_2dim.py)
    x_d, x_m_d, y_m_d, mask1_d, mask2_d = x_init_dim2(mu, nu, p=2, n=n)
    grad_d = grad_dim2(x_m_d, y_m_d, mask1_d, mask2_d, p=2, c=c_dense)
    sum_d = np.sum(grad_d * x_d)

    # Compact (FW_2dim_p2.py)
    x_c, x_m_c, y_m_c, mask1_c, mask2_c = x_init_dim2_p2(mu, nu, n)
    grad_c = grad_dim2_p2(x_m_c, y_m_c, mask1_c, mask2_c, n)
    sum_c = np.sum(grad_c * x_c)

    np.set_printoptions(precision=6, suppress=True, linewidth=160, threshold=10_000)

    for k in range(max_iter):
        lmo_d = LMO_dim2(x_d, grad_d, M, eps)
        (comp_FW, full_FW), (comp_AFW, full_AFW) = LMO_dim2_p2(x_c, grad_c, M, eps)

        gap_d = gap_calc_dim2(grad_d, lmo_d, M, sum_d)
        gap_c = gap_calc_dim2_p2(grad_c, comp_FW, M, sum_c)

        gamma_d = dense_step_size(x_d, x_m_d, y_m_d, grad_d, mu, nu, M, lmo_d, c_dense)
        gamma_c = compact_step_size(x_c, x_m_c, y_m_c, mu, nu, M, comp_FW, full_FW, comp_AFW, full_AFW)

        cost_d = UOT_cost(x_d, x_m_d, y_m_d, c_dense, mu, nu, p=2)
        cost_c = cost_dim2_p2(x_c, x_m_c, y_m_c, mu, nu)

        x_c_dense = to_dense_dim2_p2(x_c, n)
        grad_c_dense = to_dense_dim2_p2(grad_c, n)

        print("\n" + "=" * 120)
        print(f"ITERATION {k}")
        print("=" * 120)

        print("\n[DENSE FW_2dim]")
        print("LMO:", lmo_d)
        print("gap:", gap_d)
        print("stepsize:", gamma_d)
        print("sum_term:", sum_d)
        print("actual_cost:", cost_d)
        print("x (dense):\n", x_d)
        print("grad (dense):\n", grad_d)

        print("\n[COMPACT FW_2dim_p2 -> DENSE VIEW]")
        print("LMO:", format_lmo_compact(((comp_FW, full_FW), (comp_AFW, full_AFW))))
        print("gap:", gap_c)
        print("stepsize:", gamma_c)
        print("sum_term:", sum_c)
        print("actual_cost:", cost_c)
        print("x (dense):\n", x_c_dense)
        print("grad (dense):\n", grad_c_dense)

        print("\n[DIFFERENCES: dense - compact]")
        print("gap diff:", gap_d - gap_c)
        print("stepsize diff:", gamma_d - gamma_c)
        print("sum_term diff:", sum_d - sum_c)
        print("cost diff:", cost_d - cost_c)
        print("||x_diff||_inf:", np.max(np.abs(x_d - x_c_dense)))
        print("||grad_diff||_inf:", np.max(np.abs(grad_d - grad_c_dense)))

        stop_d = (gap_d <= delta) or (lmo_d == ((-1, -1, -1, -1), (-1, -1, -1, -1)))
        stop_c = (gap_c <= delta) or (full_FW == (-1, -1, -1, -1) and full_AFW == (-1, -1, -1, -1))

        if stop_d and stop_c:
            print("\nBoth solvers satisfy stopping criteria.\n")
            break

        # Remove old contributions from sum_term
        (x1FW_d, x2FW_d, y1FW_d, y2FW_d), (x1AFW_d, x2AFW_d, y1AFW_d, y2AFW_d) = lmo_d
        target_coords = {(y1FW_d, y2FW_d), (y1AFW_d, y2AFW_d)} - {(-1, -1)}
        source_coords = {(x1FW_d, x2FW_d), (x1AFW_d, x2AFW_d)} - {(-1, -1)}

        sum_d = update_sum_term_dim2(
            sum_d, grad_d, x_d, mask1_d, mask2_d, source_coords, target_coords, sign=-1
        )
        sum_c = update_sum_term_dim2_p2(
            sum_c, grad_c, x_c, full_FW, full_AFW, n, sign=-1
        )

        # Apply steps
        x_d, x_m_d, y_m_d = apply_step_dim2(x_d, x_m_d, y_m_d, grad_d, mu, nu, M, lmo_d, c_dense, p=2)
        x_c, x_m_c, y_m_c = apply_step_dim2_p2(
            x_c, x_m_c, y_m_c, mu, nu, M, comp_FW, full_FW, comp_AFW, full_AFW
        )

        # Gradient updates
        grad_d = grad_update_dim2(x_m_d, y_m_d, grad_d, mask1_d, mask2_d, c_dense, lmo_d, p=2)
        grad_c = grad_update_dim2_p2(x_m_c, y_m_c, grad_c, mask1_c, mask2_c, full_FW, full_AFW)

        # Add new contributions to sum_term
        sum_d = update_sum_term_dim2(
            sum_d, grad_d, x_d, mask1_d, mask2_d, source_coords, target_coords, sign=1
        )
        sum_c = update_sum_term_dim2_p2(
            sum_c, grad_c, x_c, full_FW, full_AFW, n, sign=1
        )


def main():
    parser = argparse.ArgumentParser(
        description="Iteration-by-iteration comparison between FW_2dim and FW_2dim_p2"
    )
    parser.add_argument("max_iter", nargs="?", type=int, default=20, help="maximum iterations (positional)")
    parser.add_argument("delta", nargs="?", type=float, default=1e-2, help="stopping gap threshold (positional)")
    parser.add_argument("eps", nargs="?", type=float, default=1e-3, help="LMO tolerance (positional)")
    parser.add_argument("--max-iter", dest="max_iter", type=int, help="maximum iterations")
    parser.add_argument("--delta", dest="delta", type=float, help="stopping gap threshold")
    parser.add_argument("--eps", dest="eps", type=float, help="LMO tolerance")
    
    args = parser.parse_args()

    mu = np.array([[0, 1], [0.2, 0.8]])
    nu = np.array([[1, 0.5], [2, 0.2]])
    M = 2 * (np.sum(mu) + np.sum(nu))

    print("mu:\n", mu)
    print("nu:\n", nu)

    compare_iteration_by_iteration(
        mu=mu,
        nu=nu,
        M=M,
        max_iter=args.max_iter,
        delta=args.delta,
        eps=args.eps,
    )


if __name__ == "__main__":
    main()
