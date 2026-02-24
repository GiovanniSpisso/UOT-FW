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
  x: transportation plan (assumed x > 0 or x == 0)
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
Compute the total UOT cost (for p=2)
Parameters:
  pi: transportation plan
  x_marg, y_marg: X and Y marginals of the transportation plan
  mu, nu: measures
'''
def cost_p2(pi, x_marg, y_marg, mu, nu):
  # pi is a 3n vector: [upper_diag | main_diag | lower_diag]
  # cost is 1 on upper/lower diagonals and 0 on main diagonal
  n = x_marg.shape[0]
  C1 = np.sum(pi[:n]) + np.sum(pi[2*n:3*n])

  mask_x = (mu != 0)
  mask_y = (nu != 0)
  # Compute entropy only on non-zero measure indices
  cost_row = np.sum(mu[mask_x] * Up(x_marg[mask_x], 2))
  cost_col = np.sum(nu[mask_y] * Up(y_marg[mask_y], 2))

  C2 = cost_row + cost_col
  return C1 + C2


'''
Utilities for 3n vector representation of banded matrix
Vector layout: [upper_diag (n) | main_diag (n) | lower_diag (n)]
Constraint: upper_diag[0] = 0, lower_diag[n-1] = 0
'''
def vec_to_mat_p2(vec, n):
    '''
    Convert 3n vector to full n x n matrix.
    vec[0:n] = upper diagonal, vec[n:2n] = main diagonal, vec[2n:3n] = lower diagonal
    '''
    matrix = np.zeros((n, n))
    
    # Main diagonal
    np.fill_diagonal(matrix, vec[n:2*n])
    
    # Upper diagonal
    np.fill_diagonal(matrix[:n-1, 1:], vec[1:n])
    
    # Lower diagonal
    np.fill_diagonal(matrix[1:, :-1], vec[2*n:3*n-1])
    
    return matrix


def vec_i_to_mat_i_p2(idx, n):
    '''
    Convert 3n vector index to matrix indices (i, j).
    '''
    if idx < n:
        # Upper diagonal: vec[idx] -> matrix[idx-1, idx]
        # vec[1] -> (0,1), vec[2] -> (1,2), ..., vec[n-1] -> (n-2, n-1)
        # vec[0] is unused (constrained to 0)
        return (idx-1, idx)
    elif idx < 2*n:
        # Main diagonal: vec[idx] -> matrix[idx-n, idx-n]
        diag_idx = idx - n
        return (diag_idx, diag_idx)
    else:
        # Lower diagonal: vec[idx] -> matrix[idx-2n+1, idx-2n]
        # vec[2n] -> (1,0), vec[2n+1] -> (2,1), ..., vec[3n-2] -> (n-1, n-2)
        # vec[3n-1] is unused (constrained to 0)
        diag_idx = idx - 2*n + 1
        return (diag_idx, diag_idx-1)


def mat_i_to_vec_i_p2(i, j, n):
    '''
    Convert matrix indices (i, j) to 3n vector index.
    Returns None if (i, j) is outside the banded structure.
    '''
    k = j - i
    return (1 - k) * n + i


'''
Initial transportation plan (for p=2)
Parameters:
  mu, nu: measures
  n: sample points
'''
def x_init_p2(mu, nu, n):
  x = np.zeros(3*n)  # 3n vector: [upper_diag (n) | main_diag (n) | lower_diag (n)]
  x_marg = np.zeros(n)
  y_marg = np.zeros(n)

  mask1 = (mu != 0)
  mask2 = (nu != 0)
  mask = mask1 & mask2

  # Compute main diagonal values (indices n to 2n-1 of the 3n vector)
  diag_vals = np.zeros(n)
  
  diag_vals[mask] = 2 * mu[mask] * nu[mask] / (mu[mask] + nu[mask])
  
  x[n:2*n] = diag_vals  # Set main diagonal part
  
  x_marg[mask] = diag_vals[mask] / mu[mask]
  y_marg[mask] = diag_vals[mask] / nu[mask]

  return x, x_marg, y_marg, mask1, mask2


'''
Function to define the gradient of UOT (for p=2)
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  mask1, mask2: masks for the gradient
  n: sample points
'''
def grad_p2(x_marg, y_marg, mask1, mask2, n):
  # Compute derivatives only where masks are true
  dx = np.zeros(n)
  dy = np.zeros(n)
  dx[mask1] = dUp_dx(x_marg[mask1], 2)
  dy[mask2] = dUp_dx(y_marg[mask2], 2)
  
  # Initialize 3n gradient vector
  grad = np.zeros(3*n)
  
  # Main diagonal: grad[n+i] corresponds to (i,i)
  # Cost: c[i,i] = 0, so grad = 0 + dx[i] + dy[i]
  m = mask1 & mask2
  grad[n:2*n][m] = dx[m] + dy[m]
  
  # Upper diagonal: grad[i+1] corresponds to (i, i+1)
  # Cost: c[i,i+1] = 1
  mask_upper = mask1[:-1] & mask2[1:]
  grad[1:n][mask_upper] = 1 + dx[:-1][mask_upper] + dy[1:][mask_upper]
  
  # Lower diagonal: grad[2n+i] corresponds to (i, i-1)
  # Cost: c[i,i-1] = 1
  mask_lower = mask1[1:] & mask2[:-1]
  grad[2*n:3*n-1][mask_lower] = 1 + dx[1:][mask_lower] + dy[:-1][mask_lower]
  
  return grad


'''
Function to find the search direction
Parameters:
  pi: transportation plan
  grad: gradient of UOT
  M: upper bound for generalized simplex
  eps: tolerance
'''
def LMO_p2(pi, grad, M, eps):
  # Frank-Wolfe direction
  idx = np.argmin(grad)
  min_val = grad[idx]
  if min_val < -eps:
    FW = idx
  else:
    FW = -1

  # Away Frank-Wolfe direction
  mask = (pi > 0)

  if not np.any(mask):
    return (FW, -1)
  else:
    grad_masked = np.where(mask, grad, -np.inf)

    max_val = grad_masked.max()
    if (max_val <= eps):
      if (pi.sum() < M):
        return (FW, -1)
      else:
        print("M: ", M, ", pi.sum(): ", pi.sum(), ". Increase M!")

    AFW = np.argmax(grad_masked)

  return (FW, AFW)


'''
Parameters:
  grad: gradient of UOT
  v: indices of the search direction
  M: upper bound for generalized simplex
  sum_term: pre-computed sum of grad_UOT * xk
'''
def gap_calc_p2(grad, v, M, sum_term):
  if v[0] != -1:
      gap = - M * grad[v[0]] + sum_term
  else:
      gap = sum_term
  return gap


'''
Optimal stepsize (p = 2)
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  mu, nu: measures
  coords: pre-computed matrix coordinates (FW_i, FW_j, AFW_i, AFW_j)
'''
def opt_step(x_marg, y_marg, mu, nu, coords):
  FW_i, FW_j, AFW_i, AFW_j = coords

  def cost_ij(ii, jj):
    return 0.0 if ii == jj else 1.0

  if FW_i == -1:
    return (cost_ij(AFW_i, AFW_j) + y_marg[AFW_j] + x_marg[AFW_i] - 2) / (1/mu[AFW_i] + 1/nu[AFW_j])
  elif AFW_i == -1:
    return (2 - cost_ij(FW_i, FW_j) - y_marg[FW_j] - x_marg[FW_i]) / (1/mu[FW_i] + 1/nu[FW_j])
  elif FW_i == AFW_i:
    return (cost_ij(FW_i, AFW_j) - cost_ij(FW_i, FW_j) + y_marg[AFW_j] - y_marg[FW_j]) / (1/nu[AFW_j] + 1/nu[FW_j])
  elif FW_j == AFW_j:
    return (cost_ij(AFW_i, FW_j) - cost_ij(FW_i, FW_j) + x_marg[AFW_i] - x_marg[FW_i]) / (1/mu[AFW_i] + 1/mu[FW_i])
  else:
    return (cost_ij(AFW_i, AFW_j) - cost_ij(FW_i, FW_j) + y_marg[AFW_j] - y_marg[FW_j] + x_marg[AFW_i]
            - x_marg[FW_i]) / (1/mu[AFW_i] + 1/mu[FW_i] + 1/nu[AFW_j] + 1/nu[FW_j])


'''
Armijo stepsize
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  grad: gradient of UOT (3n vector)
  mu, nu: measures
  v: search direction (FW_idx_3n, AFW_idx_3n) where indices are in 3n vector space
  p: main parameter that defines the p-entropy
  coords: pre-computed matrix coordinates (FW_i, FW_j, AFW_i, AFW_j)
  theta, beta, gamma: parameters for the Armijo stepsize
'''
def armijo_p2(x_marg, y_marg, grad, mu, nu, v, p, coords, theta = 1, beta = 0.4, gamma = 0.5):
  # get the 3n indices and pre-computed matrix coordinates
  FW, AFW = v
  FW_i, FW_j, AFW_i, AFW_j = coords

  def cost_ij(ii, jj):
    return 0.0 if ii == jj else 1.0

  if FW_i != -1:
    inner = grad[FW]
    if AFW_i != -1:
      inner -= grad[AFW]
      diff = (theta * (cost_ij(FW_i, FW_j) - cost_ij(AFW_i, AFW_j)) + (Up(x_marg[FW_i] + theta/mu[FW_i], p) - Up(x_marg[FW_i], p))*mu[FW_i] +
              (Up(y_marg[FW_j] + theta/nu[FW_j], p) - Up(y_marg[FW_j], p))*nu[FW_j] + (Up(x_marg[AFW_i] - theta/mu[AFW_i], p) - Up(x_marg[AFW_i], p))*mu[AFW_i]
              + (Up(y_marg[AFW_j] - theta/nu[AFW_j], p) - Up(y_marg[AFW_j], p))*nu[AFW_j])
      while diff > beta*theta*inner:
        theta = gamma * theta
        diff = (theta * (cost_ij(FW_i, FW_j) - cost_ij(AFW_i, AFW_j)) + (Up(x_marg[FW_i] + theta/mu[FW_i], p) - Up(x_marg[FW_i], p))*mu[FW_i] +
                (Up(y_marg[FW_j] + theta/nu[FW_j], p) - Up(y_marg[FW_j], p))*nu[FW_j] + (Up(x_marg[AFW_i] - theta/mu[AFW_i], p) - Up(x_marg[AFW_i], p))*mu[AFW_i]
                + (Up(y_marg[AFW_j] - theta/nu[AFW_j], p) - Up(y_marg[AFW_j], p))*nu[AFW_j])
    else:
      diff = (theta*cost_ij(FW_i, FW_j) + (Up(x_marg[FW_i] + theta/mu[FW_i], p) - Up(x_marg[FW_i], p))*mu[FW_i] +
              (Up(y_marg[FW_j] + theta/nu[FW_j], p) - Up(y_marg[FW_j], p))*nu[FW_j])
      while diff > beta*theta*inner:
        theta = gamma * theta
        diff = (theta*cost_ij(FW_i, FW_j) + (Up(x_marg[FW_i] + theta/mu[FW_i], p) - Up(x_marg[FW_i], p))*mu[FW_i] +
                (Up(y_marg[FW_j] + theta/nu[FW_j], p) - Up(y_marg[FW_j], p))*nu[FW_j])

  elif AFW_i != -1:
      inner = -grad[AFW]
      diff = (- theta * cost_ij(AFW_i, AFW_j) + (Up(x_marg[AFW_i] - theta/mu[AFW_i], p) - Up(x_marg[AFW_i], p))*mu[AFW_i]
              + (Up(y_marg[AFW_j] - theta/nu[AFW_j], p) - Up(y_marg[AFW_j], p))*nu[AFW_j])
      while diff > beta*theta*inner:
        theta = gamma * theta
        diff = (- theta * cost_ij(AFW_i, AFW_j) + (Up(x_marg[AFW_i] - theta/mu[AFW_i], p) - Up(x_marg[AFW_i], p))*mu[AFW_i]
                + (Up(y_marg[AFW_j] - theta/nu[AFW_j], p) - Up(y_marg[AFW_j], p))*nu[AFW_j])

  return theta


'''
Function to update the gradient of UOT (3n vector representation)
Parameters:
  x_marg, y_marg  : X and Y marginals of the transportation plan
  grad            : gradient of UOT (3n vector)
  mask1, mask2    : masks for the gradient (mu != 0 and nu != 0)
  coords          : pre-computed matrix coordinates (FW_i, FW_j, AFW_i, AFW_j)
'''
def update_grad_p2(x_marg, y_marg, grad, mask1, mask2, coords):
  n = x_marg.shape[0]
  p = 2  # Since this function is specific to p=2
  
  FW_i, FW_j, AFW_i, AFW_j = coords
  
  def update_row(i):
    '''Update gradient entries in row i (main, upper, lower diagonals)'''
    dUp_x = dUp_dx(x_marg[i], p)
    
    # (i, i) main diagonal at index n+i
    if mask2[i]:
      grad[n + i] = dUp_dx(y_marg[i], p) + dUp_x
    
    # (i, i+1) upper diagonal at index i+1
    if (i + 1 < n) and mask2[i + 1]:
      grad[i + 1] = 1 + dUp_dx(y_marg[i + 1], p) + dUp_x
    
    # (i, i-1) lower diagonal at index 2n+i
    if (i > 0) and mask2[i - 1]:
      grad[2*n + i - 1] = 1 + dUp_dx(y_marg[i - 1], p) + dUp_x
  
  def update_col(j):
    '''Update gradient entries in column j (main, upper, lower diagonals)'''
    dUp_y = dUp_dx(y_marg[j], p)
    
    # (j, j) main diagonal at index n+j
    if mask1[j]:
       grad[n + j] = dUp_y + dUp_dx(x_marg[j], p)
    
    # (j-1, j) upper diagonal at index j
    if (j > 0) and mask1[j - 1]:
       grad[j] = 1 + dUp_y + dUp_dx(x_marg[j - 1], p)
    
    # (j+1, j) lower diagonal at index 2n+j
    if (j + 1 < n) and mask1[j + 1]:
       grad[2*n + j] = 1 + dUp_y + dUp_dx(x_marg[j + 1], p)
  
  # Update FW row/column if needed
  if FW_i != -1 and FW_j != -1:
    update_row(FW_i)
    update_col(FW_j)
  
  # Update AFW row/column if needed
  if AFW_i != -1 and AFW_j != -1:
    update_row(AFW_i)
    update_col(AFW_j)
  
  return grad


'''
Update sum_term by adding/subtracting contributions from affected rows/columns
Parameters:
  sum_term: current sum term
  grad_xk: gradient vector (3n)
  xk: current transportation plan vector (3n)
  vk: indices from vk (filter out -1)
  n: size parameter (for converting row/col indices to 3n vector indices)
  sign: +1 to add contributions, -1 to subtract contributions
'''
def update_sum_term_p2(sum_term, grad_xk, xk, vk, n, sign):
    coord_set = set()
    for v in vk:
       if v != -1:
          # col terms
          i = v%n
          if 0 < i: coord_set.update([i, i+n])
          if i < n-1: coord_set.add(i + 2*n)

          # row terms
          j = v%(n-1)
          if 1 < j: coord_set.update([j, j+n-1, j+2*n-2])
          elif j == 0: coord_set.update([n-1, 2*n-2, 3*n-3])
          elif v == 1 or v == n: coord_set.update([1, n])
          else: coord_set.update([2*n-1, 3*n-2])

    for idx in coord_set:
       sum_term += sign * grad_xk[idx] * xk[idx]

    return sum_term


'''
Apply step update for either AFW (Away Frank-Wolfe) or FW (Frank-Wolfe) direction
Parameters:
  xk, x_marg, y_marg: transportation plan (3n vector) and marginals (n vectors)
  mu, nu: measures
  M: upper bound for generalized simplex
  vk: search direction indices (FW_idx_3n, AFW_idx_3n)
  coords: matrix coordinates (FW_i, FW_j, AFW_i, AFW_j)
Returns:
  updated xk, x_marg, y_marg
'''
def apply_step_p2(xk, x_marg, y_marg, mu, nu, M, vk, coords):
    FW, AFW = vk  # 3n vector indices
    FW_i, FW_j, AFW_i, AFW_j = coords  # matrix coordinates
    
    if AFW_i != -1:
      gamma0 = xk[AFW] - 1e-10
      gammak = min(opt_step(x_marg, y_marg, mu, nu, coords=coords), gamma0)
      xk[AFW] -= gammak
      x_marg[AFW_i] -= gammak / mu[AFW_i]
      y_marg[AFW_j] -= gammak / nu[AFW_j]
      if FW_i != -1:
        xk[FW] += gammak
        x_marg[FW_i] += gammak / mu[FW_i]
        y_marg[FW_j] += gammak / nu[FW_j]
    else:
      gamma0 = M - np.sum(xk) + xk[FW] - 1e-10
      gammak = min(opt_step(x_marg, y_marg, mu, nu, coords=coords), gamma0)
      xk[FW] += gammak
      x_marg[FW_i] += gammak / mu[FW_i]
      y_marg[FW_j] += gammak / nu[FW_j]
    
    return xk, x_marg, y_marg


'''
Pairwise Frank-Wolfe (specific for p = 2)
Parameters:
  mu, nu: measures
  M: upper bound for generalized simplex
  max_iter: max iterations
  delta, eps: tolerance
'''
def PW_FW_dim1_p2(mu, nu, M,
                  max_iter = 100, delta = 0.01, eps = 0.001):
  n = np.shape(mu)[0]

  # transportation plan, marginals and gradient initialization
  xk, x_marg, y_marg, mask1, mask2 = x_init_p2(mu, nu, n)
  grad_xk = grad_p2(x_marg, y_marg, mask1, mask2, n)
  
  # Initialize sum_term for efficient gap calculation
  sum_term = np.sum(grad_xk * xk)

  for k in range(max_iter):
    # search direction (returns 3n vector indices)
    vk = LMO_p2(xk, grad_xk, M, eps)

    # gap calculation
    gap = gap_calc_p2(grad_xk, vk, M, sum_term)

    if (gap <= delta) or (vk == (-1, -1)):
      print("FW_1dim_p2 converged after: ", k, " iterations ")
      return xk, grad_xk, x_marg, y_marg

    # Convert 3n indices to matrix coordinates (done once per iteration)
    FW, AFW = vk
    FW_i, FW_j = (-1, -1)
    AFW_i, AFW_j = (-1, -1)
    
    if FW != -1:
      result = vec_i_to_mat_i_p2(FW, n)
      if result is not None:
        FW_i, FW_j = result
    
    if AFW != -1:
      result = vec_i_to_mat_i_p2(AFW, n)
      if result is not None:
        AFW_i, AFW_j = result

    # Remove contributions from affected rows/columns before gradient update
    sum_term = update_sum_term_p2(sum_term, grad_xk, xk, vk, n, sign=-1)
    
    # Apply step update
    xk, x_marg, y_marg = apply_step_p2(xk, x_marg, y_marg, mu, nu, M, vk, coords=(FW_i, FW_j, AFW_i, AFW_j))

    # gradient update - pass both 3n indices and coordinates
    grad_xk = update_grad_p2(x_marg, y_marg, grad_xk, mask1, mask2, coords=(FW_i, FW_j, AFW_i, AFW_j))
    
    # Add back contributions from affected rows/columns after gradient update
    sum_term = update_sum_term_p2(sum_term, grad_xk, xk, vk, n, sign=+1)

  print("FW_1dim_p2 converged after: ", max_iter, " iterations ")
  return xk, grad_xk, x_marg, y_marg