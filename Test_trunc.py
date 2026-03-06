import numpy as np
from FW_1dim_trunc import PW_FW_dim1_trunc, truncated_cost, UOT_cost_upper, vector_to_matrix

# ──────────────────────────────────────────────
# Def function to calculate the gradient from a given point
# ──────────────────────────────────────────────
def dUp_dx(x, p):
    x = np.maximum(x, 0)  # clamp negatives, but assume caller passes valid data
    
    # For x == 0: return 0 (limit of derivative)
    if p == 1:
        result = np.zeros_like(x, dtype=float)
        mask_nonzero = (x > 0)
        result[mask_nonzero] = np.log(x[mask_nonzero])
    elif p > 1:
        result = (x**(p - 1) - 1) / (p - 1)
    else: # p < 1
        result = np.zeros_like(x, dtype=float)
        mask_nonzero = (x > 0)
        result[mask_nonzero] = (x[mask_nonzero]**(p-1) - 1) / (p - 1)
    
    return result


def grad_calc_from_scratch(x_marg, y_marg, s_i, s_j, c_trunc, p, n, R):    
    gx = np.zeros_like(c_trunc, dtype=float)
    pos = 0
    for k in range(-R + 1, R):
        m = n - abs(k)

        if k >= 0:
            i = np.arange(m)
            j = i + k
        else:
            j = np.arange(m)
            i = j - k

        gx[pos:pos + m] = abs(k) + dUp_dx(x_marg[i] + s_i[i], p) + dUp_dx(y_marg[j] + s_j[j], p)
        pos += m

    gs = (np.zeros(n), np.zeros(n))
            
    for k in range(n):
        gs[0][k] = 1/2 * R + dUp_dx(x_marg[k] + s_i[k], p)
        gs[1][k] = 1/2 * R + dUp_dx(y_marg[k] + s_j[k], p)
        
    return gx, gs




# ──────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────
n        = 1000
max_iter = 30000
R        = 10
p        = 1
delta    = 0.001
eps      = 0.001

seed     = 0      # set once for reproducible but different runs

rng = np.random.default_rng(seed)
mu = rng.integers(1, 1000, size=n)
nu = rng.integers(1, 1000, size=n)
M  = n * (np.sum(mu) + np.sum(nu))

print("Running UOT trunc.")
xk, (gx, gs), x_marg, y_marg, s_i, s_j = PW_FW_dim1_trunc(
    mu, nu, M, p, R, max_iter=max_iter, delta=delta, eps=eps)

c       = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
c_trunc = np.concatenate([np.full(n - abs(k), abs(k)) for k in range(-R + 1, R)])
cost = truncated_cost(xk, x_marg, y_marg, c_trunc, mu, nu, p, s_i, s_j, R)
print("cost: ", cost)

# The marginals are consistent
#x_marg_scratch = vector_to_matrix(xk, n, R).sum(axis=1)
#y_marg_scratch = vector_to_matrix(xk, n, R).sum(axis=0)
#print("Difference between x_marg and x_marg_scratch: ", np.max(np.abs(x_marg*mu - x_marg_scratch)))
#print("Difference between y_marg and y_marg_scratch: ", np.max(np.abs(y_marg*nu - y_marg_scratch)))

# The gradient is consistent
#gx_scratch, gs_scratch = grad_calc_from_scratch(x_marg_scratch/mu, y_marg_scratch/nu, s_i, s_j, c_trunc, p, n, R)
#print("Difference between gx and gx_scratch: ", np.max(np.abs(gx - gx_scratch)))
#print("Difference between gs and gs_scratch (x): ", gs[0] - gs_scratch[0])
#print("Difference between gs and gs_scratch (y): ", gs[1] - gs_scratch[1])

print(xk[:100])
print(gx[:100])