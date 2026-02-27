import numpy as np

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
        result = np.ones_like(x, dtype=float)
        mask_nonzero = (x > 0)
        result[mask_nonzero] = x[mask_nonzero] * np.log(x[mask_nonzero]) - x[mask_nonzero] + 1
    elif p == 0:
        result = np.ones_like(x, dtype=float)
        mask_nonzero = (x > 0)
        result[mask_nonzero] = x[mask_nonzero] - 1 - np.log(x[mask_nonzero])
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
    else:
        result = (x**(p - 1) - 1) / (p - 1)
    
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
  mask1 = np.ma.getmaskarray(mu)
  mask2 = np.ma.getmaskarray(nu)
  mask = mask1[:, np.newaxis] | mask2[np.newaxis, :]
  x = np.ma.array(np.zeros((n, n)), mask=mask)

  if p == 2:
    diag = 2 * mu * nu / (mu + nu)
  elif p > 1:  
    diag = ((mu * nu) / (mu**(p-1) + nu**(p-1))**(1/(p-1))) * 2**(1/(p-1))
  elif p == 1:
    diag = np.sqrt(mu * nu)
  else: # p < 1
    diag = ((mu**(p-1) + nu**(p-1)) / (2 * (mu**(p-1) * nu**(p-1))))**(1/(1-p))
  
  np.fill_diagonal(x, diag)
  x_marg = np.ma.array(diag.filled(0) / mu, mask=mask1)
  y_marg = np.ma.array(diag.filled(0) / nu, mask=mask2)

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
Optimal stepsize (p = 2)
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  c: cost function
  mu, nu: measures
  i: indices of search direction
'''
def opt_step(x_marg, y_marg, c, mu, nu, i):
  FW_i, FW_j, AFW_i, AFW_j = i
  if FW_i == -1:
    return (c[AFW_i, AFW_j] + y_marg[AFW_j] + x_marg[AFW_i] - 2) / (1/mu[AFW_i] + 1/nu[AFW_j])
  elif AFW_i == -1:
    return (2 - c[FW_i, FW_j] - y_marg[FW_j] - x_marg[FW_i]) / (1/mu[FW_i] + 1/nu[FW_j])
  elif FW_i == AFW_i:
    return (c[FW_i, AFW_j] - c[FW_i, FW_j] + y_marg[AFW_j] - y_marg[FW_j]) / (1/nu[AFW_j] + 1/nu[FW_j])
  elif FW_j == AFW_j:
    return (c[AFW_i, FW_j] - c[FW_i, FW_j] + x_marg[AFW_i] - x_marg[FW_i]) / (1/mu[AFW_i] + 1/mu[FW_i])
  else:
    return (c[AFW_i, AFW_j] - c[FW_i, FW_j] + y_marg[AFW_j] - y_marg[FW_j] + x_marg[AFW_i]
            - x_marg[FW_i]) / (1/mu[AFW_i] + 1/mu[FW_i] + 1/nu[AFW_j] + 1/nu[FW_j])

'''
Armijo stepsize
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  grad_UOT: gradient of UOT
  mu, nu: measures
  v: search direction
  c: cost function
  p: main parameter that defines the p-entropy
  theta, beta, gamma: parameters for the Armijo stepsize
'''
def armijo(x_marg, y_marg, grad_UOT, mu, nu, v, c, p, theta = 1, beta = 0.4, gamma = 0.5):
  # get the indices of the selected FW and AFW vertices
  FW_i, FW_j, AFW_i, AFW_j = v

  if FW_i != -1:
    inner = grad_UOT[FW_i, FW_j]
    if AFW_i != -1:
      inner -= grad_UOT[AFW_i, AFW_j]
      diff = (theta * (c[FW_i, FW_j] - c[AFW_i, AFW_j]) + (Up(x_marg[FW_i] + theta/mu[FW_i], p) - Up(x_marg[FW_i], p))*mu[FW_i] +
              (Up(y_marg[FW_j] + theta/nu[FW_j], p) - Up(y_marg[FW_j], p))*nu[FW_j] + (Up(x_marg[AFW_i] - theta/mu[AFW_i], p) - Up(x_marg[AFW_i], p))*mu[AFW_i]
              + (Up(y_marg[AFW_j] - theta/nu[AFW_j], p) - Up(y_marg[AFW_j], p))*nu[AFW_j])
      while diff > beta*theta*inner:
        theta = gamma * theta
        diff = (theta * (c[FW_i, FW_j] - c[AFW_i, AFW_j]) + (Up(x_marg[FW_i] + theta/mu[FW_i], p) - Up(x_marg[FW_i], p))*mu[FW_i] +
                (Up(y_marg[FW_j] + theta/nu[FW_j], p) - Up(y_marg[FW_j], p))*nu[FW_j] + (Up(x_marg[AFW_i] - theta/mu[AFW_i], p) - Up(x_marg[AFW_i], p))*mu[AFW_i]
                + (Up(y_marg[AFW_j] - theta/nu[AFW_j], p) - Up(y_marg[AFW_j], p))*nu[AFW_j])
    else:
      diff = (theta*c[FW_i, FW_j] + (Up(x_marg[FW_i] + theta/mu[FW_i], p) - Up(x_marg[FW_i], p))*mu[FW_i] +
              (Up(y_marg[FW_j] + theta/nu[FW_j], p) - Up(y_marg[FW_j], p))*nu[FW_j])
      while diff > beta*theta*inner:
        theta = gamma * theta
        diff = (theta*c[FW_i, FW_j] + (Up(x_marg[FW_i] + theta/mu[FW_i], p) - Up(x_marg[FW_i], p))*mu[FW_i] +
                (Up(y_marg[FW_j] + theta/nu[FW_j], p) - Up(y_marg[FW_j], p))*nu[FW_j])

  elif AFW_i != -1:
      inner = -grad_UOT[AFW_i, AFW_j]
      diff = (- theta * c[AFW_i, AFW_j] + (Up(x_marg[AFW_i] - theta/mu[AFW_i], p) - Up(x_marg[AFW_i], p))*mu[AFW_i]
              + (Up(y_marg[AFW_j] - theta/nu[AFW_j], p) - Up(y_marg[AFW_j], p))*nu[AFW_j])
      while diff > beta*theta*inner:
        theta = gamma * theta
        diff = (- theta * c[AFW_i, AFW_j] + (Up(x_marg[AFW_i] - theta/mu[AFW_i], p) - Up(x_marg[AFW_i], p))*mu[AFW_i]
                + (Up(y_marg[AFW_j] - theta/nu[AFW_j], p) - Up(y_marg[AFW_j], p))*nu[AFW_j])

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
def step_calc(x_marg, y_marg, grad_UOT, mu, nu, v, c, p, theta = 1, beta = 0.4, gamma = 0.5):
  if p == 2:
    return min(opt_step(x_marg, y_marg, c, mu, nu, v), theta)
  else:
    return armijo(x_marg, y_marg, grad_UOT, mu, nu, v, c, p, theta = theta, beta = beta, gamma = gamma)


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
    sum_term += sign * np.ma.dot(grad_xk[i, :], xk[i, :])
  for j in cols:
    sum_term += sign * np.ma.dot(grad_xk[:, j], xk[:, j])
  # Add back intersection (entries subtracted or added twice)
  for i in rows:
      for j in cols:
          sum_term -= sign * grad_xk[i, j] * xk[i, j]

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
    gamma0 = xk[AFW_i, AFW_j] - 1e-10
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
    gamma0 = M - np.sum(xk) + xk[FW_i, FW_j]
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
  M: upper bound for generalized simplex
  p: main parameter that defines the p-entropy
  c: cost function
  max_iter: max iterations
  delta, eps: tolerance
'''
def PW_FW_dim1(mu, nu, M, p, c,
               max_iter = 100, delta = 0.01, eps = 0.001):
  n = np.shape(mu)[0]
  # Mask zero entries in mu and nu to deal with measures with zero mass
  mu = np.ma.masked_equal(mu, 0)
  nu = np.ma.masked_equal(nu, 0)

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