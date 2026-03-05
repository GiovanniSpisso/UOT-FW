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
n        = 50
R        = 3
p        = 1
max_iter = 10000
delta    = 0.001
eps      = 0.001

m_runs   = 1
seed     = 0

# Full 2D ground cost c(i,j,k,l) = sqrt((i-k)^2 + (j-l)^2)
idx = np.arange(n)
di = (idx[:, None] - idx[None, :]) ** 2
dj = (idx[:, None] - idx[None, :]) ** 2
c2d = np.sqrt(di[:, None, :, None] + dj[None, :, None, :])   # shape (n,n,n,n)

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
    mu = rng.integers(1, 100, size=(n, n))
    nu = rng.integers(1, 100, size=(n, n))
    M  = n * (np.sum(mu) + np.sum(nu))
    return mu, nu, M

# ──────────────────────────────────────────────
# Solver wrappers
# Each returns a unified dict with keys:
#   cost, time, plan, x_marg, y_marg, extras
# ──────────────────────────────────────────────
def run_FW_2dim(mu, nu, M):
    t0 = time.time()
    xk, grad, x_marg, y_marg = PW_FW_dim2(
        mu, nu, M, p, c2d, max_iter=max_iter, delta=delta, eps=eps
    )
    elapsed = time.time() - t0
    cost = UOT_cost(xk, x_marg, y_marg, c2d, mu, nu, p)
    return dict(cost=cost, time=elapsed, plan=xk, x_marg=x_marg, y_marg=y_marg, extras={})

def run_FW_2dim_p2(mu, nu, M):
    t0 = time.time()
    xk, grad, x_marg, y_marg = PW_FW_dim2_p2(
        mu, nu, M, max_iter=max_iter, delta=delta, eps=eps
    )
    elapsed = time.time() - t0
    cost = cost_dim2_p2(xk, x_marg, y_marg, mu, nu)
    return dict(cost=cost, time=elapsed, plan=xk, x_marg=x_marg, y_marg=y_marg, extras={})

def run_FW_2dim_trunc(mu, nu, M):
    t0 = time.time()
    xk, (gx, gs), x_marg, y_marg, s_i, s_j = PW_FW_dim2_trunc(
        mu, nu, M, p, R, max_iter=max_iter, delta=delta, eps=eps
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

def run_POT(mu, nu, M):
    t0 = time.time()
    a = mu.ravel()
    b = nu.ravel()
    result = ot.solve_sample(
        X_a, X_b, a, b,
        unbalanced_type="KL",
        metric="euclidean",
        unbalanced=1
    )
    elapsed = time.time() - t0

    plan_flat = result.plan
    plan_4d   = plan_flat.reshape(n, n, n, n)

    x_marg = np.sum(plan_4d, axis=(2, 3)) / mu
    y_marg = np.sum(plan_4d, axis=(0, 1)) / nu

    cost = UOT_cost(plan_4d, x_marg, y_marg, c2d, mu, nu, p)
    return dict(
        cost=cost,
        time=elapsed,
        plan=plan_4d,
        x_marg=x_marg,
        y_marg=y_marg,
        extras=dict(pot_value=float(result.value)),
    )

# ──────────────────────────────────────────────
# SELECT WHICH SOLVERS TO RUN
# Comment out any line to skip that solver entirely.
# ──────────────────────────────────────────────
solvers = {
    "FW_2dim_p2": run_FW_2dim_p2,
    "FW_2dim": run_FW_2dim,
    #"FW_2dim_trunc": run_FW_2dim_trunc,
    #"POT": run_POT,
}

# stats[name] = {'costs': [...], 'times': [...]}
stats = {name: {"costs": [], "times": []} for name in solvers}

rng = np.random.default_rng(seed)

# ──────────────────────────────────────────────
# RUN m TIMES
# ──────────────────────────────────────────────
for run_id in range(1, m_runs + 1):
    mu, nu, M = sample_measures(rng)
    print(f"\nRun {run_id}/{m_runs}")

    for name, solver in solvers.items():
        r = solver(mu, nu, M)
        stats[name]["costs"].append(r["cost"])
        stats[name]["times"].append(r["time"])

        extra = ""
        if name == "FW_2dim_trunc":
            extra = (
                f"   upper={r['extras']['cost_upper']:.6f}"
                f"   sum(s_i*mu)={r['extras']['si_mu_sum']:.6f}"
                f"   sum(s_j*nu)={r['extras']['sj_nu_sum']:.6f}"
                f"   nnz={r['extras']['nnz']}"
            )
        elif name == "POT":
            extra = f"   pot_value={r['extras']['pot_value']:.6f}"

        print(f"  {name:<20} cost={r['cost']:.6f}   time={r['time']:.4f} s{extra}")

# ──────────────────────────────────────────────
# PRINT MEANS
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("MEAN COSTS AND TIMES")
print("=" * 60)

for name, v in stats.items():
    mean_cost = float(np.mean(v["costs"]))
    mean_time = float(np.mean(v["times"]))
    print(f"{name:<20} mean cost={mean_cost:.6f}   mean time={mean_time:.4f} s")