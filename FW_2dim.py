import numpy as np

'''
Power-like entropy function
Parameters:
  x: transportation plan
  p: main parameter that defines the p-entropy
'''
def Up(x, p):
    x = np.asarray(x)
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
    x = np.asarray(x)
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
  c: cost function
  mu, nu: measures
  p: main parameter that defines the p-entropy
"""
def UOT_cost(pi, x_marg, y_marg, c, mu, nu, p):
  C1 = np.sum(c * pi)
  cost_row, cost_col = 0, 0

  cost_row = np.sum(mu * Up(x_marg, p))
  cost_col = np.sum(nu * Up(y_marg, p))

  C2 = cost_row + cost_col
  return C1 + C2


'''
Function to define the transportation plan
Parameters:
  mu, nu: measures
  p: main parameter that defines the p-entropy
  n: sample points
'''
def x_init_dim2(mu, nu, p, n):
    den = mu + nu
    x = np.zeros((n, n, n, n))
    x_marg = np.zeros((n, n))
    y_marg = np.zeros((n, n))

    mask1 = (mu != 0)
    mask2 = (nu != 0)
    mask = mask1 & mask2

    if p == 2:
        vals = 2 * mu[mask] * nu[mask] / den[mask]

    elif p == 1:
        vals = np.sqrt(mu[mask] * nu[mask])

    elif p < 1:
        vals = (
            (mu[mask]**(p-1) + nu[mask]**(p-1)) /
            (2 * (mu[mask]**(p-1) * nu[mask]**(p-1)))
        )**(1/(1-p))

    else:  # p > 1
        vals = (mu[mask] * nu[mask]) / (mu[mask]**(p-1) + nu[mask]**(p-1))**(1/(p-1)) * 2**(1/(p-1))

    x[mask, mask] = vals

    # Marginals
    x_marg[mask] = vals / mu[mask]
    y_marg[mask] = vals / nu[mask]

    return x, x_marg, y_marg, mask1, mask2


'''
Function to define the gradient of UOT
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  mask1, mask2: mask for the zero coordinates
  p: main parameter that defines the p-entropy
  c: cost function
'''
def grad_dim2(x_marg, y_marg, mask1, mask2, p, c):
    # Compute the gradients separately
    dx = dUp_dx(x_marg, p)  # shape (n, n)
    dy = dUp_dx(y_marg, p)  # shape (n, n)

    # Add separable gradient terms
    grad_UOT = c + dx[:, :, None, None] + dy[None, None, :, :]

    # Apply masks
    mask_i_j = mask1[:, :, None, None]   # broadcast mask1 over k,l
    mask_k_l = mask2[None, None, :, :]   # broadcast mask2 over i,j
    mask = mask_i_j & mask_k_l

    grad_UOT *= mask  # zero out entries where mask is False

    return grad_UOT


'''
Function to find the search direction
Parameters:
  pi: transportation plan
  grad_UOT: gradient of UOT
  M: upper bound for generalized simplex
  eps: tolerance
'''
def LMO_dim2(pi, grad_UOT, M, eps = 0.001):
  # Frank-Wolfe direction
  flat_idx = np.argmin(grad_UOT)
  min_val = grad_UOT.flat[flat_idx]
  if min_val < -eps:
     i_FW = np.unravel_index(flat_idx, grad_UOT.shape)
  else:
     i_FW = (-1, -1, -1, -1)

  # Away Frank-Wolfe direction
  mask = pi > 0
  if not np.any(mask):
    return i_FW, (-1, -1, -1, -1)
  else:
    grad_masked = np.where(mask, grad_UOT, -np.inf)

    max_val = grad_masked.max()
    if (max_val <= eps):
       if (pi.sum() < M):
          return i_FW, (-1, -1, -1, -1)
       else:
          print("M: ", M, ", pi.sum(): ", pi.sum(), ". Increase M!")
    
    i_AFW = np.unravel_index(np.argmax(grad_masked), grad_UOT.shape)
      
  return i_FW, i_AFW


'''
Parameters:
  grad_UOT: gradient of UOT
  dir: indices of the search direction
  M: upper bound for generalized simplex
  sum_term: pre-computed sum of grad * xk
'''
def gap_calc_dim2(grad, dir, M, sum_term):
    i_FW, _ = dir

    if i_FW[0] != -1:
        gap = - M * grad[i_FW] + sum_term
    else:
        gap = sum_term

    return gap


'''
Optimal stepsize (p = 2)
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  c: cost function
  mu, nu: measures
  i: indices of the selected FW and AFW vertices
'''
def opt_step_dim2(x_marg, y_marg, c, mu, nu, i):
  (x1FW, x2FW, y1FW, y2FW), (x1AFW, x2AFW, y1AFW, y2AFW) = i
  if x1FW == -1:
    return (c[x1AFW, x2AFW, y1AFW, y2AFW] + y_marg[y1AFW, y2AFW] +
            x_marg[x1AFW, x2AFW] - 2) / (1/mu[x1AFW, x2AFW] + 1/nu[y1AFW, y2AFW])
  elif x1AFW == -1:
    return (2 - c[x1FW, x2FW, y1FW, y2FW] - y_marg[y1FW, y2FW] -
            x_marg[x1FW, x2FW]) / (1/mu[x1FW, x2FW] + 1/nu[y1FW, y2FW])
  elif (x1FW, x2FW) == (x1AFW, x2AFW):
    return (c[x1FW, x2FW, y1AFW, y2AFW] - c[x1FW, x2FW, y1FW, y2FW] + y_marg[y1AFW, y2AFW] -
            y_marg[y1FW, y2FW]) / (1/nu[y1AFW, y2AFW] + 1/nu[y1FW, y2FW])
  elif (y1FW, y2FW) == (y1AFW, y2AFW):
    return (c[x1AFW, x2AFW, y1FW, y2FW] - c[x1FW, x2FW, y1FW, y2FW] + x_marg[x1AFW, x2AFW] -
            x_marg[x1FW, x2FW]) / (1/mu[x1AFW, x2AFW] + 1/mu[x1FW, x2FW])
  else:
    return (c[x1AFW, x2AFW, y1AFW, y2AFW] - c[x1FW, x2FW, y1FW, y2FW] +
            y_marg[y1AFW, y2AFW] - y_marg[y1FW, y2FW] + x_marg[x1AFW, x2AFW] -
            x_marg[x1FW, x2FW]) / (1/mu[x1AFW, x2AFW] + 1/mu[x1FW, x2FW] + 1/nu[y1AFW, y2AFW] + 1/nu[y1FW, y2FW])


'''
Armijo stepsize
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  grad_UOT: gradient of UOT
  mu, nu: measures
  v: indices of the selected FW and AFW vertices
  c: cost function
  p: main parameter that defines the p-entropy
  theta, beta, gamma: parameters for the Armijo stepsize
'''
def armijo_dim2(x_marg, y_marg, grad_UOT, mu, nu, v, c, p, theta=1, beta=0.4, gamma=0.5):
    # get the indices of the selected FW and AFW vertices
    (x1FW, x2FW, y1FW, y2FW), (x1AFW, x2AFW, y1AFW, y2AFW) = v

    if x1FW != -1:
        inner = grad_UOT[x1FW, x2FW, y1FW, y2FW]
        if x1AFW != -1:
            inner -= grad_UOT[x1AFW, x2AFW, y1AFW, y2AFW]
            diff = (theta * (c[x1FW, x2FW, y1FW, y2FW] - c[x1AFW, x2AFW, y1AFW, y2AFW]) + 
                    (Up(x_marg[x1FW, x2FW] + theta/mu[x1FW, x2FW], p) - Up(x_marg[x1FW, x2FW], p)) * mu[x1FW, x2FW] +
                    (Up(y_marg[y1FW, y2FW] + theta/nu[y1FW, y2FW], p) - Up(y_marg[y1FW, y2FW], p)) * nu[y1FW, y2FW] + 
                    (Up(x_marg[x1AFW, x2AFW] - theta/mu[x1AFW, x2AFW], p) - Up(x_marg[x1AFW, x2AFW], p)) * mu[x1AFW, x2AFW] +
                    (Up(y_marg[y1AFW, y2AFW] - theta/nu[y1AFW, y2AFW], p) - Up(y_marg[y1AFW, y2AFW], p)) * nu[y1AFW, y2AFW])
            
            while diff > beta * theta * inner:
                theta = gamma * theta
                diff = (theta * (c[x1FW, x2FW, y1FW, y2FW] - c[x1AFW, x2AFW, y1AFW, y2AFW]) + 
                        (Up(x_marg[x1FW, x2FW] + theta/mu[x1FW, x2FW], p) - Up(x_marg[x1FW, x2FW], p)) * mu[x1FW, x2FW] +
                        (Up(y_marg[y1FW, y2FW] + theta/nu[y1FW, y2FW], p) - Up(y_marg[y1FW, y2FW], p)) * nu[y1FW, y2FW] + 
                        (Up(x_marg[x1AFW, x2AFW] - theta/mu[x1AFW, x2AFW], p) - Up(x_marg[x1AFW, x2AFW], p)) * mu[x1AFW, x2AFW] +
                        (Up(y_marg[y1AFW, y2AFW] - theta/nu[y1AFW, y2AFW], p) - Up(y_marg[y1AFW, y2AFW], p)) * nu[y1AFW, y2AFW])
        else:
            diff = (theta * c[x1FW, x2FW, y1FW, y2FW] + 
                    (Up(x_marg[x1FW, x2FW] + theta/mu[x1FW, x2FW], p) - Up(x_marg[x1FW, x2FW], p)) * mu[x1FW, x2FW] +
                    (Up(y_marg[y1FW, y2FW] + theta/nu[y1FW, y2FW], p) - Up(y_marg[y1FW, y2FW], p)) * nu[y1FW, y2FW])
            
            while diff > beta * theta * inner:
                theta = gamma * theta
                diff = (theta * c[x1FW, x2FW, y1FW, y2FW] + 
                        (Up(x_marg[x1FW, x2FW] + theta/mu[x1FW, x2FW], p) - Up(x_marg[x1FW, x2FW], p)) * mu[x1FW, x2FW] +
                        (Up(y_marg[y1FW, y2FW] + theta/nu[y1FW, y2FW], p) - Up(y_marg[y1FW, y2FW], p)) * nu[y1FW, y2FW])

    elif x1AFW != -1:
        inner = -grad_UOT[x1AFW, x2AFW, y1AFW, y2AFW]
        diff = (-theta * c[x1AFW, x2AFW, y1AFW, y2AFW] + 
                (Up(x_marg[x1AFW, x2AFW] - theta/mu[x1AFW, x2AFW], p) - Up(x_marg[x1AFW, x2AFW], p)) * mu[x1AFW, x2AFW] +
                (Up(y_marg[y1AFW, y2AFW] - theta/nu[y1AFW, y2AFW], p) - Up(y_marg[y1AFW, y2AFW], p)) * nu[y1AFW, y2AFW])
        
        while diff > beta * theta * inner:
            theta = gamma * theta
            diff = (-theta * c[x1AFW, x2AFW, y1AFW, y2AFW] + 
                    (Up(x_marg[x1AFW, x2AFW] - theta/mu[x1AFW, x2AFW], p) - Up(x_marg[x1AFW, x2AFW], p)) * mu[x1AFW, x2AFW] +
                    (Up(y_marg[y1AFW, y2AFW] - theta/nu[y1AFW, y2AFW], p) - Up(y_marg[y1AFW, y2AFW], p)) * nu[y1AFW, y2AFW])

    return theta


'''
Stepsize calculation
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  grad: gradient of UOT
  mu, nu: measures
  v: indices of the selected FW and AFW vertices
  c: cost function
  p: main parameter
  theta, beta, gamma: Armijo parameters
'''
def step_calc_dim2(x_marg, y_marg, grad_UOT, mu, nu, v, c, p, theta = 1, beta = 0.4, gamma = 0.5):
  if p == 2:
    return min(opt_step_dim2(x_marg, y_marg, c, mu, nu, v), theta)
  else:
    return armijo_dim2(x_marg, y_marg, grad_UOT, mu, nu, v, c, p, theta = theta, beta = beta, gamma = gamma)


"""
Update the gradient
Parameters:
  x_marg, y_marg  : X and Y marginals of the transportation plan
  grad_UOT        : gradient of UOT
  mask1,mask2     : masks for the zero coordinates
  c               : cost function
  v               : ((x1FW,x2FW,y1FW,y2FW), (x1AFW,x2AFW,y1AFW,y2AFW))
  p               : main parameter that defines the p-entropy
  """
def grad_update_dim2(x_marg, y_marg, grad_UOT, mask1, mask2, c, v, p):
    (x1FW, x2FW, y1FW, y2FW), (x1AFW, x2AFW, y1AFW, y2AFW) = v
    if x1FW != -1:
        # Update all entries where source is (x1FW, x2FW)
        grad_UOT[x1FW, x2FW, mask2] = (
            c[x1FW, x2FW, mask2] + 
            dUp_dx(y_marg[mask2], p) + 
            dUp_dx(x_marg[x1FW, x2FW], p))
        # Update all entries where target is (y1FW, y2FW)
        grad_UOT[mask1, y1FW, y2FW] = (
            c[mask1, y1FW, y2FW] + 
            dUp_dx(y_marg[y1FW, y2FW], p) + 
            dUp_dx(x_marg[mask1], p))
    if x1AFW != -1:
        # Update all entries where source is (x1AFW, x2AFW)
        grad_UOT[x1AFW, x2AFW, mask2] = (
            c[x1AFW, x2AFW, mask2] + 
            dUp_dx(y_marg[mask2], p) + 
            dUp_dx(x_marg[x1AFW, x2AFW], p))
        # Update all entries where target is (y1AFW, y2AFW)
        grad_UOT[mask1, y1AFW, y2AFW] = (
            c[mask1, y1AFW, y2AFW] + 
            dUp_dx(y_marg[y1AFW, y2AFW], p) + 
            dUp_dx(x_marg[mask1], p))
    
    return grad_UOT


"""
Update sum term for 4D tensor (n, n, n, n) with 2D masks.
Parameters:
  sum_term: Current sum value
  grad_xk: Gradient tensor (n, n, n, n)
  xk: Current plan tensor (n, n, n, n)
  mask1: Boolean mask for source coordinates (n, n)
  mask2: Boolean mask for target coordinates (n, n)
  rows: Set of affected source coordinate pairs {(x1, x2), ...}
  cols: Set of affected target coordinate pairs {(y1, y2), ...}
  sign: +1 or -1
"""
def update_sum_term_dim2(sum_term, grad_xk, xk, mask1, mask2, rows, cols, sign=1):
    # Update contributions for affected target coordinates (y1, y2)
    for y1, y2 in cols:
        sum_term += sign * np.sum(grad_xk[mask1, y1, y2] * xk[mask1, y1, y2])

    # Update contributions for affected source coordinates (x1, x2)
    for x1, x2 in rows:
        sum_term += sign * np.sum(grad_xk[x1, x2, mask2] * xk[x1, x2, mask2])

    # Remove double-counted intersections
    for x1, x2 in rows:
        for y1, y2 in cols:
            sum_term -= sign * grad_xk[x1, x2, y1, y2] * xk[x1, x2, y1, y2]

    return sum_term


'''
Update xk, x_marg, y_marg according to the computed step size and search direction
Parameters:
  xk: current transportation plan
  x_marg, y_marg: current marginals
  grad_xk: current gradient
  mu, nu: measures
  M: upper bound for generalized simplex
  v: search direction (FW_i, FW_j, AFW_i, AFW_j)
  c: cost function
  p: main parameter that defines the p-entropy
'''
def apply_step_dim2(xk, x_marg, y_marg, grad_xk, mu, nu, M, v, c, p):
  (x1FW, x2FW, y1FW, y2FW), (x1AFW, x2AFW, y1AFW, y2AFW) = v

  if x1AFW != -1:
    theta = xk[x1AFW, x2AFW, y1AFW, y2AFW]
    gammak = step_calc_dim2(x_marg, y_marg, grad_xk, mu, nu, v, c, p, theta=theta)

    xk[x1AFW, x2AFW, y1AFW, y2AFW] -= gammak
    x_marg[x1AFW, x2AFW] -= gammak / mu[x1AFW, x2AFW]
    y_marg[y1AFW, y2AFW] -= gammak / nu[y1AFW, y2AFW]

    if x1FW != -1:
      xk[x1FW, x2FW, y1FW, y2FW] += gammak
      x_marg[x1FW, x2FW] += gammak / mu[x1FW, x2FW]
      y_marg[y1FW, y2FW] += gammak / nu[y1FW, y2FW]
  else:
    theta = M - np.sum(xk) + xk[x1FW, x2FW, y1FW, y2FW]
    gammak = step_calc_dim2(x_marg, y_marg, grad_xk, mu, nu, v, c, p, theta=theta)

    xk[x1FW, x2FW, y1FW, y2FW] += gammak
    x_marg[x1FW, x2FW] += gammak / mu[x1FW, x2FW]
    y_marg[y1FW, y2FW] += gammak / nu[y1FW, y2FW]

  return xk, x_marg, y_marg


'''
Pairwise Frank-Wolfe
Parameters:
  mu, nu: measures
  M: upper bound for generalized simplex
  p: main parameter that defines the p-entropy
  step: stepsize calculation method
  max_iter: max iterations
  delta, eps: tolerance
'''
def PW_FW_dim2(mu, nu, M, p, c,
               max_iter = 100, delta = 0.01, eps = 0.001):
  n = np.shape(mu)[0]
  # transportation plan, marginals and gradient initialization
  xk, x_marg, y_marg, mask1, mask2 = x_init_dim2(mu, nu, p, n)
  grad_xk = grad_dim2(x_marg, y_marg, mask1, mask2, p, c)

  # Initialize sum_term for efficient gap calculation
  sum_term = np.sum(grad_xk * xk)

  for k in range(max_iter):
    # LMO calculation
    vk = LMO_dim2(xk, grad_xk, M, eps)

    # gap calculation
    gap = gap_calc_dim2(grad_xk, vk, M, sum_term)

    if (gap <= delta) or (vk == ((-1,-1,-1,-1),(-1,-1,-1,-1))):
      print("FW_2dim converged after: ", k, " iterations ")
      return (xk, grad_xk, x_marg, y_marg)

    # coordinates + rows and columns update
    (x1FW, x2FW, y1FW, y2FW), (x1AFW, x2AFW, y1AFW, y2AFW) = vk
    
    # Collect affected target and source coordinates
    target_coords = set([(y1FW, y2FW), (y1AFW, y2AFW)]) - {(-1, -1)}
    source_coords = set([(x1FW, x2FW), (x1AFW, x2AFW)]) - {(-1, -1)}
    
    # Remove contributions before gradient update
    sum_term = update_sum_term_dim2(sum_term, grad_xk, xk, mask1, mask2,
                               source_coords, target_coords, sign=-1)

    # Apply step update
    xk, x_marg, y_marg = apply_step_dim2(xk, x_marg, y_marg, grad_xk, mu, nu, M, vk, c, p)

    # gradient update
    grad_xk = grad_update_dim2(x_marg, y_marg, grad_xk, mask1, mask2, c, vk, p)

    # Add back contributions after gradient update
    sum_term = update_sum_term_dim2(sum_term, grad_xk, xk, mask1, mask2,
                               source_coords, target_coords, sign=+1)

  
  print("FW_2dim reached max iterations: ", max_iter)
  return (xk, grad_xk, x_marg, y_marg)