import ot
import time
import numpy as np
from FW_1dim       import PW_FW_dim1, UOT_cost
from FW_1dim_p2    import PW_FW_dim1_p2, cost_p2
from FW_1dim_p1_5  import PW_FW_dim1_p1_5, cost_p1_5
from FW_1dim_trunc import PW_FW_dim1_trunc, truncated_cost, UOT_cost_upper

np.set_printoptions(precision=3, suppress=True)

# ──────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────
n        = 500
max_iter = 30000
R        = 10
p        = 1
delta    = 0.001
eps      = 0.001

m_runs   = 1      # number of runs
seed     = 0      # set once for reproducible but different runs

c       = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
c_trunc = np.concatenate([np.full(n - abs(k), abs(k)) for k in range(-R + 1, R)])

X_a = np.arange(n, dtype=np.float64).reshape(-1, 1)
X_b = X_a.copy()

# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────
def sample_measures(rng):
    mu = rng.integers(1, 1000, size=n)
    nu = rng.integers(1, 1000, size=n)
    M  = n * (np.sum(mu) + np.sum(nu))
    return mu, nu, M

# ──────────────────────────────────────────────
# Solver wrappers
# Each returns a unified dict with keys:
#   cost, time, plan, x_marg, y_marg, extras (solver-specific)
# ──────────────────────────────────────────────
def run_FW_1dim(mu, nu, M):
    t0 = time.time()
    xk, grad, x_marg, y_marg = PW_FW_dim1(mu, nu, M, p, c,
                                          max_iter=max_iter, delta=delta, eps=eps)
    elapsed = time.time() - t0
    cost = UOT_cost(xk, x_marg, y_marg, c, mu, nu, p)
    return dict(cost=cost, time=elapsed, plan=xk, x_marg=x_marg, y_marg=y_marg,
                extras={})

def run_FW_1dim_p2(mu, nu, M):
    t0 = time.time()
    xk, grad, x_marg, y_marg = PW_FW_dim1_p2(mu, nu, M,
                                             max_iter=max_iter, delta=delta, eps=eps)
    elapsed = time.time() - t0
    cost = cost_p2(xk, x_marg, y_marg, mu, nu)
    return dict(cost=cost, time=elapsed, plan=xk, x_marg=x_marg, y_marg=y_marg,
                extras={})

def run_FW_1dim_p1_5(mu, nu, M):
    t0 = time.time()
    xk, grad, x_marg, y_marg = PW_FW_dim1_p1_5(mu, nu, M,
                                               max_iter=max_iter, delta=delta, eps=eps)
    elapsed = time.time() - t0
    cost = cost_p1_5(xk, x_marg, y_marg, mu, nu)
    return dict(cost=cost, time=elapsed, plan=xk, x_marg=x_marg, y_marg=y_marg,
                extras={})

def run_FW_1dim_trunc(mu, nu, M):
    t0 = time.time()
    xk, (gx, gs), x_marg, y_marg, s_i, s_j = PW_FW_dim1_trunc(
        mu, nu, M, p, R, max_iter=max_iter, delta=delta, eps=eps
    )
    elapsed = time.time() - t0
    cost       = truncated_cost(xk, x_marg, y_marg, c_trunc, mu, nu, p, s_i, s_j, R)
    cost_upper = UOT_cost_upper(cost, n, s_i, R, mu)
    si_mu_sum  = float(np.sum(s_i * mu))
    return dict(cost=cost, time=elapsed, plan=xk, x_marg=x_marg, y_marg=y_marg,
                extras=dict(s_i=s_i, s_j=s_j, cost_upper=cost_upper, si_mu_sum=si_mu_sum))

def run_POT(mu, nu, M):
    t0 = time.time()
    result = ot.solve_sample(X_a, X_b, mu, nu, unbalanced_type = 'KL',
                             metric='euclidean', unbalanced=1)
    elapsed = time.time() - t0
    return dict(cost=result.value, time=elapsed, plan=result.plan, x_marg=None, y_marg=None,
                extras={})

# ──────────────────────────────────────────────
# SELECT WHICH SOLVERS TO RUN
# Comment out any line to skip that solver entirely.
# ──────────────────────────────────────────────
solvers = {
    #'FW_1dim': run_FW_1dim,
    #'FW_1dim_p2': run_FW_1dim_p2,
    #'FW_1dim_p1_5': run_FW_1dim_p1_5,
    'FW_1dim_trunc': run_FW_1dim_trunc,
    'POT': run_POT,
}

# stats[name] = {'costs': [...], 'times': [...]}
stats = {name: {'costs': [], 'times': []} for name in solvers}

rng = np.random.default_rng(seed)

# ──────────────────────────────────────────────
# RUN m TIMES
# ──────────────────────────────────────────────
mu, nu, M = sample_measures(rng)
for run_id in range(1, m_runs + 1):
    print(f"\nRun {run_id}/{m_runs}")

    for name, solver in solvers.items():
        r = solver(mu, nu, M)
        stats[name]['costs'].append(r['cost'])
        stats[name]['times'].append(r['time'])

        extra = ""
        if name == 'FW_1dim_trunc':
            extra = f"   sum(s_i*mu)={r['extras']['si_mu_sum']:.6f}"

        print(f"  {name:<20} cost={r['cost']:.6f}   time={r['time']:.4f} s{extra}")


# ──────────────────────────────────────────────
# PRINT MEANS
# ──────────────────────────────────────────────
print('\n' + '='*60)
print('MEAN COSTS AND TIMES')
print('='*60)

for name, v in stats.items():
    mean_cost = float(np.mean(v['costs']))
    mean_time = float(np.mean(v['times']))
    print(f"{name:<20} mean cost={mean_cost:.6f}   mean time={mean_time:.4f} s")