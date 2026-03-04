import time
import numpy as np

from FW_1dim_trunc import (
    PW_FW_dim1_trunc as pw_new,
    truncated_cost as truncated_cost_new,
    build_c as build_c_new,
    grad_trunc as grad_trunc_new,
    LMO_x as LMO_x_new,
    vector_index_to_matrix_indices as vec2ij_new,
)
from FW_1dim_trunc_0 import (
    PW_FW_dim1_trunc as pw_old,
    truncated_cost as truncated_cost_old,
    build_c as build_c_old,
    grad_trunc as grad_trunc_old,
    LMO_x as LMO_x_old,
    vector_index_to_matrix_indices as vec2ij_old,
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


def recompute_marginals_from_xk(xk, mu, nu, n, R):
    """Rebuild row/col marginals from vectorized banded plan xk."""
    row_sum = np.zeros(n, dtype=np.float64)
    col_sum = np.zeros(n, dtype=np.float64)

    pos = 0
    for k in range(-R + 1, R):
        diag_len = n - abs(k)
        vals = xk[pos : pos + diag_len]

        if k >= 0:
            i = np.arange(0, n - k)
            j = i + k
        else:
            i = np.arange(-k, n)
            j = i + k

        row_sum[i] += vals
        col_sum[j] += vals
        pos += diag_len

    x_marg = np.divide(row_sum, mu, out=np.zeros_like(row_sum), where=(mu != 0))
    y_marg = np.divide(col_sum, nu, out=np.zeros_like(col_sum), where=(nu != 0))
    return x_marg, y_marg


def analyze_point_from_xk(
    xk, mu, nu, M, p, R, eps,
    build_c_fn, grad_fn, lmo_fn, vec2ij_fn, needs_mask=False
):
    """From xk: recompute marginals, gradient, run LMO, and collect values at LMO coords."""
    n = len(mu)
    c_trunc = build_c_fn(n, R)

    x_marg, y_marg = recompute_marginals_from_xk(xk, mu, nu, n, R)

    if needs_mask:
        mask1 = x_marg > 0
        mask2 = y_marg > 0
        grad_out = grad_fn(x_marg, y_marg, mask1, mask2, c_trunc, p, n, R)
    else:
        grad_out = grad_fn(x_marg, y_marg, c_trunc, p, n, R)

    # Expecting (grad_x, grad_s)
    grad_x = grad_out[0]
    FW_x, AFW_x = lmo_fn(xk, grad_x, M, eps)

    def coord_payload(idx):
        if idx is None or idx == -1:
            return None
        i, j = vec2ij_fn(int(idx), n, R)
        return {
            "idx": int(idx),
            "i": int(i),
            "j": int(j),
            "grad": float(grad_x[idx]),
            "x_marg": float(x_marg[i]),
            "y_marg": float(y_marg[j]),
            "xk": float(xk[idx]),
        }

    return {
        "x_marg": x_marg,
        "y_marg": y_marg,
        "grad_x": grad_x,
        "FW": coord_payload(FW_x),
        "AFW": coord_payload(AFW_x),
    }


# ──────────────────────────────────────────────
# Solver wrappers
# ──────────────────────────────────────────────
def run_fw_1dim_trunc():
    t0 = time.perf_counter()
    out = pw_new(mu.copy(), nu.copy(), M, p, R, max_iter=max_iter, delta=delta, eps=eps)
    elapsed = time.perf_counter() - t0

    if out is None:
        return dict(cost=None, time=elapsed, status="no return (max_iter reached)", extras={})

    xk, (gx, gs), x_marg_out, y_marg_out, s_i, s_j = out
    c_trunc = build_c_new(n, R)
    cost = truncated_cost_new(xk, x_marg_out, y_marg_out, c_trunc, mu, nu, p, s_i, s_j, R)

    analysis = analyze_point_from_xk(
        xk=xk, mu=mu, nu=nu, M=M, p=p, R=R, eps=eps,
        build_c_fn=build_c_new, grad_fn=grad_trunc_new, lmo_fn=LMO_x_new, vec2ij_fn=vec2ij_new,
        needs_mask=False
    )

    return dict(
        cost=float(cost),
        time=elapsed,
        status="ok",
        extras=dict(
            analysis=analysis,
            max_abs_dx=float(np.max(np.abs(analysis["x_marg"] - x_marg_out))),
            max_abs_dy=float(np.max(np.abs(analysis["y_marg"] - y_marg_out))),
        ),
    )


def run_fw_1dim_trunc_0():
    t0 = time.perf_counter()
    out = pw_old(mu.copy(), nu.copy(), M, p, R, max_iter=max_iter, delta=delta, eps=eps)
    elapsed = time.perf_counter() - t0

    if out is None:
        return dict(cost=None, time=elapsed, status="no return (max_iter reached)", extras={})

    xk, grad, x_marg_out, y_marg_out, s_i, s_j = out
    c_trunc = build_c_old(n, R)
    cost = truncated_cost_old(xk, x_marg_out, y_marg_out, c_trunc, mu, nu, p, s_i, s_j, R)

    analysis = analyze_point_from_xk(
        xk=xk, mu=mu, nu=nu, M=M, p=p, R=R, eps=eps,
        build_c_fn=build_c_old, grad_fn=grad_trunc_old, lmo_fn=LMO_x_old, vec2ij_fn=vec2ij_old,
        needs_mask=True
    )

    return dict(
        cost=float(cost),
        time=elapsed,
        status="ok",
        extras=dict(
            analysis=analysis,
            max_abs_dx=float(np.max(np.abs(analysis["x_marg"] - x_marg_out))),
            max_abs_dy=float(np.max(np.abs(analysis["y_marg"] - y_marg_out))),
        ),
    )


# ──────────────────────────────────────────────
# SELECT WHICH SOLVERS TO RUN
# Comment out one line to skip that solver.
# ──────────────────────────────────────────────
results = {}
# results["FW_1dim_trunc_0"] = run_fw_1dim_trunc_0()
results["FW_1dim_trunc"] = run_fw_1dim_trunc()

# ──────────────────────────────────────────────
# PRINT RESULTS
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("PW_FW_dim1_trunc benchmark: cost, time, and LMO diagnostics")
print("=" * 70)

for name, r in results.items():
    if r["cost"] is None:
        print(f"{name:<18} | time = {r['time']:.6f} s | cost = N/A | {r['status']}")
        continue

    print(f"{name:<18} | time = {r['time']:.6f} s | cost = {r['cost']:.10f}")

    ex = r.get("extras", {})
    if "max_abs_dx" in ex and "max_abs_dy" in ex:
        print(f"  recomputed marg diff: max|dx|={ex['max_abs_dx']:.3e}, max|dy|={ex['max_abs_dy']:.3e}")

    a = ex.get("analysis", {})
    fw = a.get("FW")
    afw = a.get("AFW")

    if fw is None:
        print("  LMO FW: no improving direction")
    else:
        print(
            f"  LMO FW idx={fw['idx']} (i,j)=({fw['i']},{fw['j']}) | "
            f"grad={fw['grad']:.6e} | x_marg={fw['x_marg']:.6e} | "
            f"y_marg={fw['y_marg']:.6e} | xk={fw['xk']:.6e}"
        )

    if afw is None:
        print("  LMO AFW: no away direction")
    else:
        print(
            f"  LMO AFW idx={afw['idx']} (i,j)=({afw['i']},{afw['j']}) | "
            f"grad={afw['grad']:.6e} | x_marg={afw['x_marg']:.6e} | "
            f"y_marg={afw['y_marg']:.6e} | xk={afw['xk']:.6e}"
        )

if "FW_1dim_trunc_0" in results and "FW_1dim_trunc" in results:
    c0 = results["FW_1dim_trunc_0"]["cost"]
    c1 = results["FW_1dim_trunc"]["cost"]
    if c0 is not None and c1 is not None:
        print("-" * 70)
        print(f"Absolute cost difference: {abs(c0 - c1):.10f}")




