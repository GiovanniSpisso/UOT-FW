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



'''
Cost function
Parameters:
  x1, x2, y1, y2: coordinates of the transportation plan
'''
def cost(x1, x2, y1, y2):
    return np.sqrt((x1 - y1)**2 + (x2 - y2)**2)


"""
Compute the total UOT cost:
Parameters:
  pi: transportation plan
  x_marg, y_marg: X and Y marginals of the transportation plan
  mu, nu: measures
  p: main parameter that defines the p-entropy
"""
def UOT_cost(pi, x_marg, y_marg, mu, nu, p):
    def transport_cost(pi):
        n = pi.shape[0]  # assume pi is (n,n,n,n)
        I, J, K, L = np.meshgrid(
            np.arange(n), np.arange(n), np.arange(n), np.arange(n), indexing="ij"
        )

        # Compute cost on all indices at once
        C = cost(I, J, K, L)

        # Multiply elementwise with pi and sum
        return np.sum(C * pi)

    def marginal_penalty(x_marg, y_marg, mu, nu, p):
        term_x = np.sum(mu * Up(x_marg, p))
        term_y = np.sum(nu * Up(y_marg, p))
        return term_x + term_y

    cost_transport = transport_cost(pi)
    cost_marginals = marginal_penalty(x_marg, y_marg, mu, nu, p)
    return cost_transport + cost_marginals


'''
Function to define the transportation plan
Parameters:
  mu, nu: measures
  p: main parameter that defines the p-entropy
  n: sample points
'''
def x_init(mu, nu, p, n):
    den = mu + nu
    x = np.zeros((n, n, n, n))
    x_marg = np.zeros((n, n))
    y_marg = np.zeros((n, n))

    mask1 = (mu != 0)
    mask2 = (nu != 0)
    mask = mask1 & mask2

    I, J = np.where(mask)

    if p == 2:
        vals = 2 * mu[I, J] * nu[I, J] / den[I, J]

    elif p == 1:
        vals = np.sqrt(mu[I, J] * nu[I, J])

    elif p < 1:
        vals = (
            (mu[I, J]**(p-1) + nu[I, J]**(p-1)) /
            (2 * (mu[I, J]**(p-1) * nu[I, J]**(p-1)))
        )**(1/(1-p))

    else:  # p > 1
        vals = (
            (mu[I, J] * nu[I, J]) /
            (mu[I, J]**(p-1) + nu[I, J]**(p-1))
        )**(1/(p-1)) * 2**(1/(p-1))

    x[I, J, I, J] = vals

    # Marginals
    x_marg[I, J] = vals / mu[I, J]
    y_marg[I, J] = vals / nu[I, J]

    return x, x_marg, y_marg, mask1, mask2


'''
Function to define the gradient of UOT
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  mask : mask for the zero coordinates
  p: main parameter that defines the p-entropy
  n: dimension
'''
def grad_init(x_marg, y_marg, mask1, mask2, p, n):
    # Compute the gradients separately
    dx = dUp_dx(x_marg, p)  # shape (n, n)
    dy = dUp_dx(y_marg, p)  # shape (n, n)

    # Create cost grid
    I, J, K, L = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), np.arange(n), indexing='ij'
    )
    C = cost(I, J, K, L)

    # Add separable gradient terms
    grad_UOT = C + dx[:, :, None, None] + dy[None, None, :, :]

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
  grad: gradient of UOT
  zero_coord: mask for the zero coordinates
  M: upper bound for generalized simplex
  eps: tolerance
'''
def direction(pi, grad, M, eps = 0.001):
  # Frank-Wolfe direction
  min_val = grad.min()
  if min_val < -eps:
    i_FW = np.unravel_index(np.argmin(grad), grad.shape)
  else:
    i_FW = (-1, -1, -1, -1)

  # Away Frank-Wolfe direction
  mask = pi > 0
  if not np.any(mask):
    return i_FW, (-1, -1, -1, -1)
  masked_grad = np.where(mask, grad, -np.inf)
  max_val = masked_grad.max()
  i_AFW = np.unravel_index(np.argmax(masked_grad), grad.shape)
  if (max_val <= eps):
    if (pi.sum() < M):
      return i_FW, (-1, -1, -1, -1)
    else:
      print("M: ", M, ", pi.sum(): ", pi.sum(), ". Increase M!")

  return i_FW, i_AFW


'''
Parameters:
  grad: gradient of UOT
  dir: indices of the search direction
  M: upper bound for generalized simplex
  sum_term: pre-computed sum of grad * xk
'''
def gap_calc(grad, dir, M, sum_term):
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
  mu, nu: measures
  i: indices of the selected FW and AFW vertices
'''
def opt_step(x_marg, y_marg, mu, nu, i):
  (x1FW, x2FW, y1FW, y2FW), (x1AFW, x2AFW, y1AFW, y2AFW) = i
  if i[0][0] == -1:
    return (cost(x1AFW, x2AFW, y1AFW, y2AFW) + y_marg[y1AFW, y2AFW] +
            x_marg[x1AFW, x2AFW] - 2) / (1/mu[x1AFW, x2AFW] + 1/nu[y1AFW, y2AFW])
  elif i[1][0] == -1:
    return (2 - cost(x1FW, x2FW, y1FW, y2FW) - y_marg[y1FW, y2FW] -
            x_marg[x1FW, x2FW]) / (1/mu[x1FW, x2FW] + 1/nu[y1FW, y2FW])
  elif i[0][:2] == i[1][:2]:
    return (cost(x1FW, x2FW, y1AFW, y2AFW) - cost(x1FW, x2FW, y1FW, y2FW) + y_marg[y1AFW, y2AFW] -
            y_marg[y1FW, y2FW]) / (1/nu[y1AFW, y2AFW] + 1/nu[y1FW, y2FW])
  elif i[0][2:] == i[1][2:]:
    return (cost(x1AFW, x2AFW, y1FW, y2FW) - cost(x1FW, x2FW, y1FW, y2FW) + x_marg[x1AFW, x2AFW] -
            x_marg[x1FW, x2FW]) / (1/mu[x1AFW, x2AFW] + 1/mu[x1FW, x2FW])
  else:
    return (cost(x1AFW, x2AFW, y1AFW, y2AFW) - cost(x1FW, x2FW, y1FW, y2FW) +
            y_marg[y1AFW, y2AFW] - y_marg[y1FW, y2FW] + x_marg[x1AFW, x2AFW] -
            x_marg[x1FW, x2FW]) / (1/mu[x1AFW, x2AFW] + 1/mu[x1FW, x2FW] + 1/nu[y1AFW, y2AFW] + 1/nu[y1FW, y2FW])


'''
Armijo stepsize
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  grad: gradient of UOT
  mu, nu: measures
  v: indices of the selected FW and AFW vertices
  p: main parameter that defines the p-entropy
  theta, beta, gamma: parameters for the Armijo stepsize
'''
def armijo(x_marg, y_marg, grad, mu, nu, v, p, theta = 1, beta = 0.4, gamma = 0.5):
  # get the indices of the selected FW and AFW vertices
  FW, AFW = v

  if FW[0] != -1:
    inner = grad[FW]
    if AFW[0] != -1:
      inner -= grad[AFW]
      diff = (theta * (cost(*FW) - cost(*AFW)) + (Up(x_marg[FW[0],FW[1]] + theta/mu[FW[0],FW[1]], p) - Up(x_marg[FW[0],FW[1]], p))*mu[FW[0],FW[1]] +
              (Up(y_marg[FW[2],FW[3]] + theta/nu[FW[2],FW[3]], p) - Up(y_marg[FW[2],FW[3]], p))*nu[FW[2],FW[3]] +
               (Up(x_marg[AFW[0],AFW[1]] - theta/mu[AFW[0],AFW[1]], p) - Up(x_marg[AFW[0],AFW[1]], p))*mu[AFW[0],AFW[1]]
              + (Up(y_marg[AFW[2],AFW[3]] - theta/nu[AFW[2],AFW[3]], p) - Up(y_marg[AFW[2],AFW[3]], p))*nu[AFW[2],AFW[3]])
      while diff > beta*theta*inner:
        theta = gamma * theta
        diff = (theta * (cost(*FW) - cost(*AFW)) + (Up(x_marg[FW[0],FW[1]] + theta/mu[FW[0],FW[1]], p) - Up(x_marg[FW[0],FW[1]], p))*mu[FW[0],FW[1]] +
                (Up(y_marg[FW[2],FW[3]] + theta/nu[FW[2],FW[3]], p) - Up(y_marg[FW[2],FW[3]], p))*nu[FW[2],FW[3]] +
                 (Up(x_marg[AFW[0],AFW[1]] - theta/mu[AFW[0],AFW[1]], p) - Up(x_marg[AFW[0],AFW[1]], p))*mu[AFW[0],AFW[1]]
                + (Up(y_marg[AFW[2],AFW[3]] - theta/nu[AFW[2],AFW[3]], p) - Up(y_marg[AFW[2],AFW[3]], p))*nu[AFW[2],AFW[3]])
    else:
      diff = (theta*cost(*FW) + (Up(x_marg[FW[0],FW[1]] + theta/mu[FW[0],FW[1]], p) - Up(x_marg[FW[0],FW[1]], p))*mu[FW[0],FW[1]] +
              (Up(y_marg[FW[2],FW[3]] + theta/nu[FW[2],FW[3]], p) - Up(y_marg[FW[2],FW[3]], p))*nu[FW[2],FW[3]])
      while diff > beta*theta*inner:
        theta = gamma * theta
        diff = (theta*cost(*FW) + (Up(x_marg[FW[0],FW[1]] + theta/mu[FW[0],FW[1]], p) - Up(x_marg[FW[0],FW[1]], p))*mu[FW[0],FW[1]] +
                (Up(y_marg[FW[2],FW[3]] + theta/nu[FW[2],FW[3]], p) - Up(y_marg[FW[2],FW[3]], p))*nu[FW[2],FW[3]])

  elif AFW[0] != -1:
      inner = -grad[AFW]
      diff = (- theta * cost(*AFW) + (Up(x_marg[AFW[0],AFW[1]] - theta/mu[AFW[0],AFW[1]], p) - Up(x_marg[AFW[0],AFW[1]], p))*mu[AFW[0],AFW[1]]
              + (Up(y_marg[AFW[2],AFW[3]] - theta/nu[AFW[2],AFW[3]], p) - Up(y_marg[AFW[2],AFW[3]], p))*nu[AFW[2],AFW[3]])
      while diff > beta*theta*inner:
        theta = gamma * theta
        diff = (- theta * cost(*AFW) + (Up(x_marg[AFW[0],AFW[1]] - theta/mu[AFW[0],AFW[1]], p) - Up(x_marg[AFW[0],AFW[1]], p))*mu[AFW[0],AFW[1]]
                + (Up(y_marg[AFW[2],AFW[3]] - theta/nu[AFW[2],AFW[3]], p) - Up(y_marg[AFW[2],AFW[3]], p))*nu[AFW[2],AFW[3]])

  return theta


'''
Stepsize calculation
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  grad: gradient of UOT
  mu, nu: measures
  v: indices of the selected FW and AFW vertices
  p: main parameter
  theta, beta, gamma: Armijo parameters
'''
def step_calc(x_marg, y_marg, grad, mu, nu, v, p, step = "armijo", theta = 1, beta = 0.4, gamma = 0.5):
  if step == "armijo":
    return armijo(x_marg, y_marg, grad, mu, nu, v, p, theta = theta, beta = beta, gamma = gamma)
  elif step == "optimal":
    return min(opt_step(x_marg, y_marg, mu, nu, v), theta)
  else:
    raise ValueError("Stepsize not recognized! Use 'armijo' or 'optimal'.")



"""
Update the gradient
Parameters:
  x_marg, y_marg  : X and Y marginals of the transportation plan
  grad            : gradient of UOT
  mask1,mask2     : masks for the zero coordinates
  v               : ((x1FW,x2FW,y1FW,y2FW), (x1AFW,x2AFW,y1AFW,y2AFW))
  p               : main parameter that defines the p-entropy
  """
def grad_update(x_marg, y_marg, grad, mask1, mask2, v, p):
    i_FW, i_AFW = v

    # Helper function for each update
    def update_XY(coords):
        if coords[0] != -1:
            idx_i, idx_j = np.nonzero(mask1)
            k, l = coords[2], coords[3]
            grad[idx_i, idx_j, k, l] = (
                cost(idx_i, idx_j, k, l) + dUp_dx(y_marg[k, l], p) + dUp_dx(x_marg[idx_i, idx_j], p)
            )
            idx_i, idx_j = np.nonzero(mask2)
            i0, j0 = coords[0], coords[1]
            grad[i0, j0, idx_i, idx_j] = (
                cost(i0, j0, idx_i, idx_j) + dUp_dx(y_marg[idx_i, idx_j], p) + dUp_dx(x_marg[i0, j0], p)
            )

    update_XY(i_FW)
    update_XY(i_AFW)

    return grad


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
def PW_FW_dim2(mu, nu, M, p, step = "armijo",
               max_iter = 100, delta = 0.01, eps = 0.001):
  n = np.shape(mu)[0]
  # transportation plan, marginals and gradient initialization
  xk, x_marg, y_marg, mask1, mask2 = x_init(mu, nu, p, n)
  grad_xk = grad_init(x_marg, y_marg, mask1, mask2, p, n)

  # Initialize sum_term for efficient gap calculation
  sum_term = np.sum(grad_xk * xk)
  mask1_idx = np.nonzero(mask1)  # (array of i, array of j)
  mask2_idx = np.nonzero(mask2)  # (array of k, array of l)

  for k in range(max_iter):
    # search direction vertices
    vk = direction(xk, grad_xk, M, eps)

    # gap calculation
    gap = gap_calc(grad_xk, vk, M, sum_term)

    if (gap <= delta) or (vk == ((-1,-1,-1,-1),(-1,-1,-1,-1))):
      print("Converged after: ", k, " iterations ")
      return (xk, grad_xk, x_marg, y_marg)

    # coordinates + rows and columns update
    (x1FW, x2FW, y1FW, y2FW), (x1AFW, x2AFW, y1AFW, y2AFW) = vk
    
    # Collect affected target and source coordinates
    target_coords = set([(y1FW, y2FW), (y1AFW, y2AFW)]) - {(-1, -1)}
    source_coords = set([(x1FW, x2FW), (x1AFW, x2AFW)]) - {(-1, -1)}
    
    # Remove contributions before gradient update
    # For each affected target (y1, y2): remove grad[:,:,y1,y2] * xk[:,:,y1,y2]
    for y1, y2 in target_coords:
      sum_term -= np.sum(grad_xk[:, :, y1, y2] * xk[:, :, y1, y2])
    
    # For each affected source (x1, x2): remove grad[x1,x2,:,:] * xk[x1,x2,:,:]
    for x1, x2 in source_coords:
      sum_term -= np.sum(grad_xk[x1, x2, :, :] * xk[x1, x2, :, :])
    
    # Add back intersection (entries subtracted twice)
    for x1, x2 in source_coords:
      for y1, y2 in target_coords:
        sum_term += grad_xk[x1, x2, y1, y2] * xk[x1, x2, y1, y2]
    
    if x1AFW != -1:
      gammak = step_calc(x_marg, y_marg, grad_xk, mu, nu, vk,
                         p = p, step = step, theta = xk[x1AFW, x2AFW, y1AFW, y2AFW])

      xk[x1AFW, x2AFW, y1AFW, y2AFW] -= gammak
      x_marg[x1AFW, x2AFW] -= gammak / mu[x1AFW, x2AFW]
      y_marg[y1AFW, y2AFW] -= gammak / nu[y1AFW, y2AFW]
      if x1FW != -1:

        xk[x1FW, x2FW, y1FW, y2FW] += gammak
        x_marg[x1FW, x2FW] += gammak / mu[x1FW, x2FW]
        y_marg[y1FW, y2FW] += gammak / nu[y1FW, y2FW]
    else:
      # stepsize
      gammak = step_calc(x_marg, y_marg, grad_xk, mu, nu, vk,
                         p = p, step = step, theta = M - np.sum(xk) + xk[x1FW, x2FW, y1FW, y2FW])

      xk[x1FW, x2FW, y1FW, y2FW] += gammak
      x_marg[x1FW, x2FW] += gammak / mu[x1FW, x2FW]
      y_marg[y1FW, y2FW] += gammak / nu[y1FW, y2FW]

    # gradient update
    grad_xk = grad_update(x_marg, y_marg, grad_xk, mask1, mask2, vk, p)

    # Add back contributions after gradient update
    # For each affected target (y1, y2): add grad[:,:,y1,y2] * xk[:,:,y1,y2]
    for y1, y2 in target_coords:
      sum_term += np.sum(grad_xk[:, :, y1, y2] * xk[:, :, y1, y2])
    
    # For each affected source (x1, x2): add grad[x1,x2,:,:] * xk[x1,x2,:,:]
    for x1, x2 in source_coords:
      sum_term += np.sum(grad_xk[x1, x2, :, :] * xk[x1, x2, :, :])
    
    # Remove intersection again (to correct for double addition)
    for x1, x2 in source_coords:
      for y1, y2 in target_coords:
        sum_term -= grad_xk[x1, x2, y1, y2] * xk[x1, x2, y1, y2]

  print("Converged after: ", max_iter, " iterations ")
  return (xk, grad_xk, x_marg, y_marg)