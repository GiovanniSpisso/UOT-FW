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
n        = 100
R        = 5
p        = 1
max_iter = 40000
delta    = 0.001
eps      = 0.001

m_runs   = 1
seed     = 0

# Full 2D ground cost c(i,j,k,l) = sqrt((i-k)^2 + (j-l)^2)
idx = np.arange(n)
di = (idx[:, None] - idx[None, :]) ** 2
dj = (idx[:, None] - idx[None, :]) ** 2
c2d = np.sqrt(di[:, None, :, None] + dj[None, :, None, :])   # shape (n,n,n,n)

# Flattened cost for POT matrix-based solvers (shape: n^2 x n^2)
C_flat = c2d.reshape(n * n, n * n)

# Truncated cost tensor
c_trunc, _ = cost_matrix_trunc_dim2(R)

# POT point clouds
xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
X_a = np.column_stack([xs.ravel(), ys.ravel()])
X_b = X_a.copy()

# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────
def sample_measures(rng):
    mu = rng.integers(1, 100, size=(n, n)).astype(float)
    nu = rng.integers(1, 100, size=(n, n)).astype(float)
    return mu, nu

def _plan4d_marginals_and_cost(plan_flat, mu, nu, p_cost):
    plan_4d = np.asarray(plan_flat, dtype=float).reshape(n, n, n, n)

    mu_safe = np.where(mu > 0, mu, 1.0)
    nu_safe = np.where(nu > 0, nu, 1.0)

    row_sum = np.sum(plan_4d, axis=(2, 3))
    col_sum = np.sum(plan_4d, axis=(0, 1))

    x_marg = np.where(mu > 0, row_sum / mu_safe, 0.0)
    y_marg = np.where(nu > 0, col_sum / nu_safe, 0.0)

    cost = UOT_cost(plan_4d, x_marg, y_marg, c2d, mu, nu, p_cost)
    return plan_4d, x_marg, y_marg, cost

# ──────────────────────────────────────────────
# Solver wrappers
# Each returns a unified dict with keys:
#   cost, time, plan, x_marg, y_marg, extras
# ──────────────────────────────────────────────
def run_FW_2dim(mu, nu):
    t0 = time.time()
    xk, grad, x_marg, y_marg = PW_FW_dim2(
        mu, nu, p, c2d, max_iter=max_iter, delta=delta, eps=eps
    )
    elapsed = time.time() - t0
    cost = UOT_cost(xk, x_marg, y_marg, c2d, mu, nu, p)
    return dict(cost=cost, time=elapsed, plan=xk, x_marg=x_marg, y_marg=y_marg, extras={})

def run_FW_2dim_p2(mu, nu):
    t0 = time.time()
    xk, grad, x_marg, y_marg = PW_FW_dim2_p2(
        mu, nu, max_iter=max_iter, delta=delta, eps=eps
    )
    elapsed = time.time() - t0
    cost = cost_dim2_p2(xk, x_marg, y_marg, mu, nu)
    return dict(cost=cost, time=elapsed, plan=xk, x_marg=x_marg, y_marg=y_marg, extras={})

def run_FW_2dim_trunc(mu, nu):
    t0 = time.time()
    xk, (gx, gs), x_marg, y_marg, s_i, s_j = PW_FW_dim2_trunc(
        mu, nu, p, R, max_iter=max_iter, delta=delta, eps=eps
    )
    elapsed = time.time() - t0
    cost       = truncated_cost_dim2(xk, x_marg, y_marg, c_trunc, mu, nu, p, s_i, s_j, R)
    cost_upper = UOT_cost_upper_dim2(cost, n, s_i, R, mu)
    si_mu_sum  = float(np.sum(s_i * mu))
    sj_nu_sum  = float(np.sum(s_j * nu))
    nnz        = int(np.sum(xk > 1e-12))
    return dict(
        cost=cost,
        time=elapsed,
        plan=xk,
        x_marg=x_marg,
        y_marg=y_marg,
        extras=dict(
            s_i=s_i,
            s_j=s_j,
            cost_upper=cost_upper,
            si_mu_sum=si_mu_sum,
            sj_nu_sum=sj_nu_sum,
            nnz=nnz,
        ),
    )

# POT: solve_sample
def run_POT_solve_sample(mu, nu):
    t0 = time.time()
    a = mu.ravel()
    b = nu.ravel()
    result = ot.solve_sample(
        X_a, X_b, a, b,
        unbalanced_type="KL",
        metric="euclidean",
        unbalanced=1,
    )
    elapsed = time.time() - t0

    plan_4d, x_marg, y_marg, cost = _plan4d_marginals_and_cost(result.plan, mu, nu, p_cost=p)
    return dict(
        cost=cost,
        time=elapsed,
        plan=plan_4d,
        x_marg=x_marg,
        y_marg=y_marg,
        extras=dict(pot_value=float(result.value)),
    )

# POT: solve on explicit cost matrix
def run_POT_solve(mu, nu):
    t0 = time.time()
    a = mu.ravel()
    b = nu.ravel()
    result = ot.solve(C_flat, a, b, unbalanced_type="KL", unbalanced=1)
    elapsed = time.time() - t0

    plan_4d, x_marg, y_marg, cost = _plan4d_marginals_and_cost(result.plan, mu, nu, p_cost=p)
    return dict(
        cost=cost,
        time=elapsed,
        plan=plan_4d,
        x_marg=x_marg,
        y_marg=y_marg,
        extras=dict(pot_value=float(result.value)),
    )

# POT: sinkhorn_unbalanced
def run_POT_sinkhorn(mu, nu):
    t0 = time.time()
    a = mu.ravel()
    b = nu.ravel()
    plan_flat = ot.unbalanced.sinkhorn_unbalanced(
        a, b, M=C_flat, reg=0.01, reg_m=1, numItermax=1000000
    )
    elapsed = time.time() - t0

    plan_4d, x_marg, y_marg, cost = _plan4d_marginals_and_cost(plan_flat, mu, nu, p_cost=1)
    return dict(cost=cost, time=elapsed, plan=plan_4d, x_marg=x_marg, y_marg=y_marg, extras={})

# POT: sinkhorn_unbalanced_translation_invariant
def run_POT_sinkhorn_translation_invariant(mu, nu):
    t0 = time.time()
    a = mu.ravel()
    b = nu.ravel()
    plan_flat = ot.unbalanced.sinkhorn_unbalanced_translation_invariant(
        a, b, M=C_flat, reg=0.01, reg_m=1, numItermax=1000000,
        translation_invariant=True, stopThr=1e-10
    )
    elapsed = time.time() - t0

    plan_4d, x_marg, y_marg, cost = _plan4d_marginals_and_cost(plan_flat, mu, nu, p_cost=1)
    return dict(cost=cost, time=elapsed, plan=plan_4d, x_marg=x_marg, y_marg=y_marg, extras={})

# POT: lbfgsb_unbalanced (p=1 style)
def run_POT_lbfgsb_p1(mu, nu):
    t0 = time.time()
    a = mu.ravel()
    b = nu.ravel()
    plan_flat = ot.unbalanced.lbfgsb_unbalanced(
        a, b,
        M=C_flat,
        reg=0,
        reg_m=1.0,
        reg_div="kl",
        regm_div="kl",
    )
    elapsed = time.time() - t0

    plan_4d, x_marg, y_marg, cost = _plan4d_marginals_and_cost(plan_flat, mu, nu, p_cost=1)
    return dict(cost=cost, time=elapsed, plan=plan_4d, x_marg=x_marg, y_marg=y_marg, extras={})

# POT: lbfgsb_unbalanced (p=2 style)
def run_POT_lbfgsb_p2(mu, nu):
    t0 = time.time()
    a = mu.ravel()
    b = nu.ravel()
    plan_flat = ot.unbalanced.lbfgsb_unbalanced(
        a, b,
        M=C_flat,
        reg=0,
        reg_m=1.0,
        reg_div="l2",
        regm_div="l2",
    )
    elapsed = time.time() - t0

    plan_4d, x_marg, y_marg, cost = _plan4d_marginals_and_cost(plan_flat, mu, nu, p_cost=2)
    return dict(cost=cost, time=elapsed, plan=plan_4d, x_marg=x_marg, y_marg=y_marg, extras={})

# POT: mm_unbalanced (p=1 style)
def run_POT_mm_unbalanced_p1(mu, nu):
    t0 = time.time()
    a = mu.ravel()
    b = nu.ravel()
    plan_flat = ot.unbalanced.mm_unbalanced(
        a, b,
        M=C_flat,
        reg_m=1.0,
        div="kl",
    )
    elapsed = time.time() - t0

    plan_4d, x_marg, y_marg, cost = _plan4d_marginals_and_cost(plan_flat, mu, nu, p_cost=1)
    return dict(cost=cost, time=elapsed, plan=plan_4d, x_marg=x_marg, y_marg=y_marg, extras={})

# POT: mm_unbalanced (p=2 style)
def run_POT_mm_unbalanced_p2(mu, nu):
    t0 = time.time()
    a = mu.ravel()
    b = nu.ravel()
    plan_flat = ot.unbalanced.mm_unbalanced(
        a, b,
        M=C_flat,
        reg_m=1.0,
        div="l2",
    )
    elapsed = time.time() - t0

    plan_4d, x_marg, y_marg, cost = _plan4d_marginals_and_cost(plan_flat, mu, nu, p_cost=2)
    return dict(cost=cost, time=elapsed, plan=plan_4d, x_marg=x_marg, y_marg=y_marg, extras={})

# Backward-compatible alias with your old function name
def run_POT(mu, nu):
    return run_POT_solve_sample(mu, nu)

# ──────────────────────────────────────────────
# SELECT WHICH SOLVERS TO RUN
# Comment out any line to skip that solver entirely.
# ──────────────────────────────────────────────
solvers = {
    #"FW_2dim_p2": run_FW_2dim_p2,
    #"FW_2dim": run_FW_2dim,
    #"FW_2dim_trunc": run_FW_2dim_trunc,
    #"POT_solve_sample": run_POT_solve_sample,
    #"POT_solve": run_POT_solve,
    "POT_sinkhorn": run_POT_sinkhorn,
    "POT_sinkhorn_translation_invariant": run_POT_sinkhorn_translation_invariant,
    #"POT_lbfgsb_p1": run_POT_lbfgsb_p1,
    #"POT_lbfgsb_p2": run_POT_lbfgsb_p2,
    #"POT_mm_unbalanced_p1": run_POT_mm_unbalanced_p1,
    #"POT_mm_unbalanced_p2": run_POT_mm_unbalanced_p2,
}

# stats[name] = {'costs': [...], 'times': [...]}
stats = {name: {"costs": [], "times": []} for name in solvers}

rng = np.random.default_rng(seed)

# ──────────────────────────────────────────────
# RUN m TIMES
# ──────────────────────────────────────────────
for run_id in range(1, m_runs + 1):
    mu, nu = sample_measures(rng)
    print(f"\nRun {run_id}/{m_runs}")

    for name, solver in solvers.items():
        r = solver(mu, nu)
        stats[name]["costs"].append(r["cost"])
        stats[name]["times"].append(r["time"])

        extra = ""
        if "cost_upper" in r["extras"]:
            extra += (
                f"   upper={r['extras']['cost_upper']:.6f}"
                f"   sum(s_i*mu)={r['extras']['si_mu_sum']:.6f}"
                f"   sum(s_j*nu)={r['extras']['sj_nu_sum']:.6f}"
                f"   nnz={r['extras']['nnz']}"
            )
        if "pot_value" in r["extras"]:
            extra += f"   pot_value={r['extras']['pot_value']:.6f}"

        print(f"  {name:<35} cost={r['cost']:.6f}   time={r['time']:.4f} s{extra}")

# ──────────────────────────────────────────────
# PRINT MEANS
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("MEAN COSTS AND TIMES")
print("=" * 60)

for name, v in stats.items():
    mean_cost = float(np.mean(v["costs"]))
    mean_time = float(np.mean(v["times"]))
    print(f"{name:<35} mean cost={mean_cost:.6f}   mean time={mean_time:.4f} s")