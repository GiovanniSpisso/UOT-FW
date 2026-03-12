import numpy as np
from itertools import combinations # for step_p1

'''
Power-like entropy function
Parameters:
  x: transportation plan
  p: main parameter that defines the p-entropy
'''
def Up(x, p):
    x = np.maximum(x, 0)  # clamp negatives, but assume caller passes valid data
    
    if p == 1:
        # For x == 0: result = 1 (limit)
        result = np.full_like(x, 10000, dtype=float)
        mask_nonzero = (x > 0)
        result[mask_nonzero] = x[mask_nonzero] * np.log(x[mask_nonzero]) - x[mask_nonzero] + 1
    else:
        result = (x**p - p * (x - 1) - 1) / (p * (p - 1))
    
    return result


'''
Derivative of power-like entropy function
Parameters:
  x: transportation plan
  p: main parameter that defines the p-entropy
'''
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


"""
Compute the total UOT cost:
Parameters:
  pi: transportation plan
  x_marg, y_marg: X and Y marginals of the transportation plan
  mu, nu: measures
  p: main parameter that defines the p-entropy
"""
def UOT_cost(pi, x_marg, y_marg, c, mu, nu, p):
  C1 = np.multiply(c, pi).sum()

  cost_row = np.sum(mu * Up(x_marg, p))
  cost_col = np.sum(nu * Up(y_marg, p))

  C2 = cost_row + cost_col
  return C1 + C2


'''
Initial transportation plan
Parameters:
  mu, nu: measures
  p: main parameter that defines the p-entropy
  n: sample points
'''
def x_init(mu, nu, p, n):
  x = np.zeros((n, n))

  if p == 2:
    diag = 2 * mu * nu / (mu + nu)
  elif p > 1:  
    diag = ((mu * nu) / (mu**(p-1) + nu**(p-1))**(1/(p-1))) * 2**(1/(p-1))
  elif p == 1:
    diag = np.sqrt(mu * nu)
  else: # p < 1
    diag = ((mu**(p-1) + nu**(p-1)) / (2 * (mu**(p-1) * nu**(p-1))))**(1/(1-p))
  
  np.fill_diagonal(x, diag)
  x_marg = diag / mu
  y_marg = diag / nu

  return x, x_marg, y_marg


'''
Function to define the gradient of UOT
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  p: main parameter that defines the p-entropy
  c: cost function
'''
def grad(x_marg, y_marg, p, c):
  dx = dUp_dx(x_marg, p)  # shape n
  dy = dUp_dx(y_marg, p)  # shape n

  # Add separable gradient terms
  grad_UOT = c + dx[:, None] + dy[None, :]

  return grad_UOT


'''
Function to find the search direction
Parameters:
  pi: transportation plan
  grad_UOT: gradient of UOT
  M: upper bound for generalized simplex
  eps: tolerance
'''
def LMO(pi, grad_UOT, M, eps):
  # Frank-Wolfe direction
  flat_idx = np.argmin(grad_UOT)
  min_val = grad_UOT.flat[flat_idx]
  if min_val < -eps:
    FW_i, FW_j = np.unravel_index(flat_idx, grad_UOT.shape)
  else:
    FW_i, FW_j = -1, -1

  # Away Frank-Wolfe direction
  mask = (pi > 0)

  if not np.any(mask):
    return (FW_i, FW_j, -1, -1)
  else:
    grad_masked = np.where(mask, grad_UOT, -np.inf)

    max_val = grad_masked.max()
    if (max_val <= eps):
      if (pi.sum() < M):
        return (FW_i, FW_j, -1, -1)
      else:
        print("M: ", M, ", pi.sum(): ", pi.sum(), ". Increase M!")

    AFW_i, AFW_j = np.unravel_index(np.argmax(grad_masked), grad_UOT.shape)

  return (FW_i, FW_j, AFW_i, AFW_j)


'''
Parameters:
  grad_UOT: gradient of UOT
  dir: indices of the search direction
  M: upper bound for generalized simplex
  sum_term: pre-computed sum of grad_UOT * xk
'''
def gap_calc(grad_UOT, dir, M, sum_term):
  if dir[0] != -1:
      gap = - M * grad_UOT[dir[0], dir[1]] + sum_term
  else:
      gap = sum_term
  return gap


'''
Function to list the coordinates to update
Parameters:
    x_marg, y_marg: X and Y marginals of the transportation plan
    grad: gradient with respect to the transport plan
    mu, nu: measures
    v: (i_FW, j_FW, i_AFW, j_AFW)
    c: cost function (vector form)
'''
def coord_updates(x_marg, y_marg, grad, mu, nu, v, c):
    FW_ix, FW_jx, AFW_ix, AFW_jx = v

    x_updates = {}  # i -> (a_i, mu_i, coeff) where coeff multiplies theta in delta: delta = coeff * theta
    y_updates = {}  # j -> (b_j, nu_j, coeff)

    def add_x(i, coeff):
        if i in x_updates:
            x0, mu0, coeff0 = x_updates[i]
            x_updates[i] = (x0, mu0, coeff0 + coeff)
        else:
            x = x_marg[i]
            x_updates[i] = (x, mu[i], coeff)

    def add_y(j, coeff):
        if j in y_updates:
            y0, nu0, coeff0 = y_updates[j]
            y_updates[j] = (y0, nu0, coeff0 + coeff)
        else:
            y = y_marg[j]
            y_updates[j] = (y, nu[j], coeff)

    # Directional derivative <grad, d> + 
    # Contributions to marginals from x-plan FW/AFW (these affect x_marg at i_* and y_marg at j_*) +
    # Contributions from supports FW/AFW +
    inner = 0.0
    cost_lin = 0.0      # Constant linear coefficient for cost term
    if FW_ix != -1: 
        inner += grad[FW_ix, FW_jx]
        add_x(FW_ix, +1.0)
        add_y(FW_jx, +1.0)
        cost_lin += c[FW_ix, FW_jx]
    if AFW_ix != -1:  
        inner -= grad[AFW_ix, AFW_jx]
        add_x(AFW_ix, -1.0)
        add_y(AFW_jx, -1.0)
        cost_lin -= c[AFW_ix, AFW_jx]

    return x_updates, y_updates, inner, cost_lin


'''
Optimal stepsize for p = 2
Parameters:
    x_updates, y_updates: coordinates to update with their current values and coefficients
    cost_lin: linear coefficient for the cost term
'''
def step_p2(x_updates, y_updates, cost_lin):
    numerator, denominator = cost_lin, 0.0
    for x, mu_i, coeff in x_updates.values():
        numerator   += coeff * (x - 1)
        denominator -= coeff**2 / mu_i
    for y, nu_j, coeff in y_updates.values():
        numerator   += coeff * (y - 1)
        denominator -= coeff**2 / nu_j

    return numerator / denominator


'''
Optimal stepsize for p = 1
Parameters:
    x_updates, y_updates: coordinates to update with their current values and coefficients
    cost_lin: linear coefficient for the cost term
'''
def step_p1(x_updates, y_updates, cost_lin):
    const, terms_num, terms_den = np.exp(-cost_lin), [], []
    constraint = np.inf
    for x, mu_i, coeff in x_updates.values():
        # divide in different cases depending on the coefficient
        if coeff == 1:
            const *= mu_i
            terms_num.append(x * mu_i)
        elif coeff == -1:
            const /= - mu_i
            terms_den.append(- x * mu_i)
            constraint = min(constraint, x * mu_i)
    for y, nu_j, coeff in y_updates.values():
        if coeff == 1:
            const *= nu_j
            terms_num.append(y * nu_j)
        elif coeff == -1:
            const /= - nu_j
            terms_den.append(- y * nu_j)
            constraint = min(constraint, y * nu_j)

    def comb(terms, k):
        if k < 0:
            return 0.0
        return sum(np.prod(combo) for combo in combinations(terms, k))

    # degree of the polynomial is max of the two sizes
    deg_num, deg_den = len(terms_num), len(terms_den)
    deg = max(deg_num, deg_den)

    coeffs = []
    for k in range(deg + 1):
        coeff = comb(terms_num, deg_num - deg + k) - const * comb(terms_den, deg_den - deg + k)
        coeffs.append(coeff)

    roots = np.roots(coeffs)  
    real_roots = roots[np.isreal(roots)].real
    step = np.max(real_roots[real_roots <= constraint])
    
    return step


'''
Armijo stepsize for truncated UOT
Parameters:
    x_updates, y_updates: coordinates to update with their current values and coefficients
    inner: directional derivative <grad, d>
    cost_lin: linear coefficient for the cost term
    p: main parameter
    theta, beta, gamma: Armijo parameters
'''
def armijo(x_updates, y_updates, inner, cost_lin, p, theta=1.0, beta=0.4, gamma=0.5):
    def obj_change(theta_val):
        diff = theta_val * (cost_lin)

        # Entropy changes for x marginals
        for _, (x, mu_i, coeff) in x_updates.items():
            d = coeff * theta_val / mu_i
            diff += (Up(x + d, p) - Up(x, p)) * mu_i

        # Entropy changes for y marginals
        for _, (y, nu_j, coeff) in y_updates.items():
            d = coeff * theta_val / nu_j
            diff += (Up(y + d, p) - Up(y, p)) * nu_j

        return diff

    # Backtracking
    diff = obj_change(theta)
    while diff > beta * theta * inner:
        assert theta > 1e-10, "Armijo stepsize became too small"

        theta *= gamma
        diff = obj_change(theta)
    
    return theta


'''
Stepsize calculation
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  grad: gradient of UOT
  mu, nu: measures
  v: search direction
  c: cost function
  p: main parameter
  theta, beta, gamma: Armijo parameters
'''
def step_calc(x_marg, y_marg, grad, mu, nu, v, c, p, theta = 1.0, beta = 0.4, gamma = 0.5):
    x_updates, y_updates, inner, cost_lin = coord_updates(x_marg, y_marg, grad, mu, nu, v, c)
    if p == 1:
        step = min(step_p1(x_updates, y_updates, cost_lin), theta)
    elif p == 2:
        step = min(step_p2(x_updates, y_updates, cost_lin), theta)
    else:
        step = armijo(x_updates, y_updates, inner, cost_lin, p, theta = theta, beta = beta, gamma = gamma)
    return step


'''
Function to update the gradient of UOT
Parameters:
  x_marg, y_marg  : X and Y marginals of the transportation plan
  grad_UOT        : gradient of UOT
  c               : cost function
  v               : indices of the search direction
  p               : main parameter that defines the p-entropy
'''
def update_grad(x_marg, y_marg, grad_UOT, c, v, p):
    FW_i, FW_j, AFW_i, AFW_j = v
    if FW_i != -1:
        grad_UOT[FW_i, :] = (c[FW_i, :] + dUp_dx(y_marg, p) + dUp_dx(x_marg[FW_i], p))
        grad_UOT[:, FW_j] = (c[:, FW_j] + dUp_dx(y_marg[FW_j], p) + dUp_dx(x_marg, p))
    if AFW_i != -1:
        grad_UOT[AFW_i, :] = (c[AFW_i, :] + dUp_dx(y_marg, p) + dUp_dx(x_marg[AFW_i], p))
        grad_UOT[:, AFW_j] = (c[:, AFW_j] + dUp_dx(y_marg[AFW_j], p) + dUp_dx(x_marg, p))
    
    return grad_UOT


'''
Update sum_term by adding/subtracting contributions from affected rows/columns
Parameters:
  sum_term: current sum term
  grad_xk: gradient vector
  xk: current transportation plan vector
  rows, cols: affected rows and columns (as sets/lists of indices)
  sign: +1 to add contributions, -1 to subtract contributions
'''
def update_sum_term(sum_term, grad_xk, xk, rows, cols, sign=1):
  for i in rows:
    sum_term += sign * np.dot(grad_xk[i, :], xk[i, :])
  for j in cols:
    sum_term += sign * np.dot(grad_xk[:, j], xk[:, j])
  # Add back intersection (entries subtracted or added twice)
  for i in rows:
      for j in cols:
          sum_term -= sign * np.dot(grad_xk[i, j], xk[i, j])

  return sum_term


'''
Update xk, x_marg, y_marg according to the computed step size and search direction
Parameters:
  xk: current transportation plan
  x_marg, y_marg: current marginals
  grad_xk: current gradient
  mu, nu: measures
  M: upper bound for generalized simplex
  vk: search direction (FW_i, FW_j, AFW_i, AFW_j)
  c: cost function
  p: main parameter that defines the p-entropy
'''
def apply_step(xk, x_marg, y_marg, grad_xk, mu, nu, M, vk, c, p):
  FW_i, FW_j, AFW_i, AFW_j = vk  # coordinates
  
  if AFW_i != -1:
    gamma0 = xk[AFW_i, AFW_j]
    # stepsize
    gammak = step_calc(x_marg, y_marg, grad_xk, mu, nu, vk, c, p, theta = gamma0)
    xk[AFW_i, AFW_j] -= gammak
    x_marg[AFW_i] -= gammak / mu[AFW_i]
    y_marg[AFW_j] -= gammak / nu[AFW_j]
    if FW_i != -1:
      xk[FW_i, FW_j] += gammak
      x_marg[FW_i] += gammak / mu[FW_i]
      y_marg[FW_j] += gammak / nu[FW_j]
  else:
    gamma0 = min(max(np.max(mu), np.max(nu)), M - np.sum(xk) + xk[FW_i, FW_j])
    # stepsize
    gammak = step_calc(x_marg, y_marg, grad_xk, mu, nu, vk, c, p, theta = gamma0)
    xk[FW_i, FW_j] += gammak
    x_marg[FW_i] += gammak / mu[FW_i]
    y_marg[FW_j] += gammak / nu[FW_j]
  
  return xk, x_marg, y_marg


'''
Pairwise Frank-Wolfe
Parameters:
  mu, nu: measures
  p: main parameter that defines the p-entropy
  c: cost function
  max_iter: max iterations
  delta, eps: tolerance
'''
def PW_FW_dim1(mu, nu, p, c,
               max_iter = 100, delta = 0.01, eps = 0.001):
  n = np.shape(mu)[0]
  M = n * (np.sum(mu) + np.sum(nu)) # upper bound for generalized simplex

  # initial transportation plan, marginals and gradient initialization
  xk, x_marg, y_marg = x_init(mu, nu, p, n)
  grad_xk = grad(x_marg, y_marg, p, c)
  
  # Initialize sum_term for efficient gap calculation
  sum_term = np.sum(grad_xk * xk)

  for k in range(max_iter):
    # search direction
    vk = LMO(xk, grad_xk, M, eps)

    # gap calculation
    gap = gap_calc(grad_xk, vk, M, sum_term)

    if (gap <= delta) or (vk == (-1,-1,-1,-1)):
      print("FW_1dim converged after: ", k, " iterations ")
      return xk, grad_xk, x_marg, y_marg

    # coordinates + rows and columns update
    FW_i, FW_j, AFW_i, AFW_j = vk

    # rows and columns update
    rows, cols = set([FW_i, AFW_i]) - {-1}, set([FW_j, AFW_j]) - {-1}

    # Remove contributions from affected rows/columns before gradient update
    sum_term = update_sum_term(sum_term, grad_xk, xk, rows, cols, sign=-1)
    
    # Apply step update
    xk, x_marg, y_marg = apply_step(xk, x_marg, y_marg, grad_xk, mu, nu, M, vk, c, p)

    # gradient update
    grad_xk = update_grad(x_marg, y_marg, grad_xk, c, vk, p)
    
    # Add back contributions from affected rows/columns after gradient update
    sum_term = update_sum_term(sum_term, grad_xk, xk, rows, cols, sign=+1)

  print("FW_1dim converged after: ", max_iter, " iterations ")
  return xk, grad_xk, x_marg, y_marg