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
n        = 1000
max_iter = 10000
R        = 5
p        = 1
delta    = 0.001
eps      = 0.001

np.random.seed(0)
mu = np.random.randint(1, 1000, size=n)
nu = np.random.randint(1, 1000, size=n)
M  = n * (np.sum(mu) + np.sum(nu))

c       = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
c_trunc = np.concatenate([np.full(n - abs(k), abs(k)) for k in range(-R + 1, R)])

X_a = np.arange(n, dtype=np.float64).reshape(-1, 1)
X_b = X_a.copy()

# ──────────────────────────────────────────────
# Solver wrappers
# Each returns a unified dict with keys:
#   cost, time, plan, x_marg, y_marg, extras (solver-specific)
# Returns None if the solver is not run.
# ──────────────────────────────────────────────

def run_FW_1dim():
    t0 = time.time()
    xk, grad, x_marg, y_marg = PW_FW_dim1(mu, nu, M, p, c,
                                           max_iter=max_iter, delta=delta, eps=eps)
    elapsed = time.time() - t0
    cost = UOT_cost(xk, x_marg, y_marg, c, mu, nu, p)
    return dict(cost=cost, time=elapsed, plan=xk, x_marg=x_marg, y_marg=y_marg,
                extras={})

def run_FW_1dim_p2():
    t0 = time.time()
    xk, grad, x_marg, y_marg = PW_FW_dim1_p2(mu, nu, M,
                                              max_iter=max_iter, delta=delta, eps=eps)
    elapsed = time.time() - t0
    cost = cost_p2(xk, x_marg, y_marg, mu, nu)
    return dict(cost=cost, time=elapsed, plan=xk, x_marg=x_marg, y_marg=y_marg,
                extras={})

def run_FW_1dim_p1_5():
    t0 = time.time()
    xk, grad, x_marg, y_marg = PW_FW_dim1_p1_5(mu, nu, M,
                                                max_iter=max_iter, delta=delta, eps=eps)
    elapsed = time.time() - t0
    cost = cost_p1_5(xk, x_marg, y_marg, mu, nu)
    return dict(cost=cost, time=elapsed, plan=xk, x_marg=x_marg, y_marg=y_marg,
                extras={})

def run_FW_1dim_trunc():
    t0 = time.time()
    xk, (gx, gs), x_marg, y_marg, s_i, s_j = PW_FW_dim1_trunc(mu, nu, M, p, R,
                                                              max_iter=max_iter, delta=delta, eps=eps)
    elapsed = time.time() - t0
    cost       = truncated_cost(xk, x_marg, y_marg, c_trunc, mu, nu, p, s_i, s_j, R)
    cost_upper = UOT_cost_upper(cost, n, s_i, R, mu)
    return dict(cost=cost, time=elapsed, plan=xk, x_marg=x_marg, y_marg=y_marg,
                extras=dict(s_i=s_i, s_j=s_j, cost_upper=cost_upper))

def run_POT():
    t0 = time.time()
    result = ot.solve_sample(X_a, X_b, mu.ravel(), nu.ravel(),
                             metric='euclidean', unbalanced=1)
    elapsed = time.time() - t0
    plan    = result.plan
    x_marg  = np.sum(plan, axis=1) / mu
    y_marg  = np.sum(plan, axis=0) / nu
    cost    = UOT_cost(plan, x_marg, y_marg, c, mu, nu, p)
    return dict(cost=cost, time=elapsed, plan=plan, x_marg=x_marg, y_marg=y_marg,
                extras=dict(pot_value=result.value))

# ──────────────────────────────────────────────
# SELECT WHICH SOLVERS TO RUN
# Comment out any line to skip that solver entirely.
# ──────────────────────────────────────────────
results = {}
#results['FW_1dim']       = run_FW_1dim()
#results['FW_1dim_p2']    = run_FW_1dim_p2()
#results['FW_1dim_p1_5']  = run_FW_1dim_p1_5()
results['FW_1dim_trunc'] = run_FW_1dim_trunc()
results['POT']           = run_POT()

# ──────────────────────────────────────────────
# PRINT RESULTS
# Each block checks if the solver was run before printing.
# ──────────────────────────────────────────────
print('\n' + '='*60)
print('COSTS')
print('='*60)
if r := results.get('FW_1dim'):
    print(f"FW_1dim cost:              {r['cost']:.6f}")
if r := results.get('FW_1dim_p2'):
    print(f"FW_1dim_p2 cost:           {r['cost']:.6f}")
if r := results.get('FW_1dim_p1_5'):
    print(f"FW_1dim_p1_5 cost:         {r['cost']:.6f}")
if r := results.get('FW_1dim_trunc'):
    print(f"FW_1dim_trunc cost:        {r['cost']:.6f}")
    print(f"FW_1dim_trunc upper bound: {r['extras']['cost_upper']:.6f}")
    print(f"S_i total: {np.sum(r['extras']['s_i'] * mu):.6f}   "
          f"S_j total: {np.sum(r['extras']['s_j'] * nu):.6f}")
if r := results.get('POT'):
    print(f"POT cost (my formula):     {r['cost']:.6f}")
    print(f"POT cost (library value):  {r['extras']['pot_value']:.6f}")

print('\n' + '='*60)
print('TIMES')
print('='*60)
for name, r in results.items():
    print(f"{name:<20} {r['time']:.4f} s")