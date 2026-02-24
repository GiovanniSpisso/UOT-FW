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
Compute the total UOT cost (for p=1.5)
Parameters:
  pi: transportation plan (7n vector)
  x_marg, y_marg: X and Y marginals of the transportation plan
  mu, nu: measures
'''
def cost_p1_5(pi, x_marg, y_marg, mu, nu):
  # pi is a 7n vector: [upper3 | upper2 | upper1 | main_diag | lower1 | lower2 | lower3]
  # cost is distance on off-diagonal, 0 on main diagonal
  n = x_marg.shape[0]
  
  # C1: transportation cost
  # Upper diagonals: distances 3, 2, 1
  C1 = 3 * np.sum(pi[:n]) + 2 * np.sum(pi[n:2*n]) + 1 * np.sum(pi[2*n:3*n])
  # Lower diagonals: distances 1, 2, 3
  C1 += 1 * np.sum(pi[4*n:5*n]) + 2 * np.sum(pi[5*n:6*n]) + 3 * np.sum(pi[6*n:7*n])

  mask_x = (mu != 0)
  mask_y = (nu != 0)
  # Compute entropy only on non-zero measure indices
  cost_row = np.sum(mu[mask_x] * Up(x_marg[mask_x], 1.5))
  cost_col = np.sum(nu[mask_y] * Up(y_marg[mask_y], 1.5))

  C2 = cost_row + cost_col
  return C1 + C2


'''
Utilities for 7n vector representation of 7-diagonal matrix
Vector layout: [upper3 (n) | upper2 (n) | upper1 (n) | main_diag (n) | lower1 (n) | lower2 (n) | lower3 (n)]
Constraints: upper3[0:3]=0, upper2[0:2]=0, upper1[0]=0, lower1[n-1]=0, lower2[n-2:n]=0, lower3[n-3:n]=0
'''
def vec_to_mat_p1_5(vec, n):
    '''
    Convert 7n vector to full n x n matrix.
    '''
    matrix = np.zeros((n, n))
    
    # Main diagonal
    np.fill_diagonal(matrix, vec[3*n:4*n])
    
    # Upper diagonals
    np.fill_diagonal(matrix[:n-1, 1:], vec[2*n+1:3*n])  # upper1 (skip vec[2n]=0)
    np.fill_diagonal(matrix[:n-2, 2:], vec[n+2:2*n])    # upper2 (skip vec[n:n+2]=0)
    np.fill_diagonal(matrix[:n-3, 3:], vec[3:n])        # upper3 (skip vec[0:3]=0)
    
    # Lower diagonals
    np.fill_diagonal(matrix[1:, :-1], vec[4*n:5*n-1])   # lower1 (skip vec[5n-1]=0)
    np.fill_diagonal(matrix[2:, :-2], vec[5*n:6*n-2])   # lower2 (skip vec[6n-2:6n]=0)
    np.fill_diagonal(matrix[3:, :-3], vec[6*n:7*n-3])   # lower3 (skip vec[7n-3:7n]=0)
    
    return matrix


def vec_i_to_mat_i_p1_5(idx, n):
    '''
    Convert 7n vector index to matrix indices (i, j).
    Returns (i, j) or None if unused/constrained entry.
    '''
    if idx < n:
        # upper3: vec[idx] -> matrix[idx-3, idx]
        # vec[3] -> (0,3), vec[4] -> (1,4), ..., vec[n-1] -> (n-4, n-1)
        # vec[0:3] are unused (constrained to 0)
        return (idx-3, idx) if idx >= 3 else None
    elif idx < 2*n:
        # upper2: vec[idx] -> matrix[idx-n-2, idx-n]
        # vec[n+2] -> (0,2), vec[n+3] -> (1,3), ..., vec[2n-1] -> (n-3, n-1)
        # vec[n:n+2] are unused (constrained to 0)
        return (idx-n-2, idx-n) if idx >= n+2 else None
    elif idx < 3*n:
        # upper1: vec[idx] -> matrix[idx-2n-1, idx-2n]
        # vec[2n+1] -> (0,1), vec[2n+2] -> (1,2), ..., vec[3n-1] -> (n-2, n-1)
        # vec[2n] is unused (constrained to 0)
        return (idx-2*n-1, idx-2*n) if idx >= 2*n+1 else None
    elif idx < 4*n:
        # main diagonal: vec[idx] -> matrix[idx-3n, idx-3n]
        return (idx-3*n, idx-3*n)
    elif idx < 5*n:
        # lower1: vec[idx] -> matrix[idx-4n+1, idx-4n]
        # vec[4n] -> (1,0), vec[4n+1] -> (2,1), ..., vec[5n-2] -> (n-1, n-2)
        # vec[5n-1] is unused (constrained to 0)
        return (idx-4*n+1, idx-4*n) if idx-4*n+1 < n else None
    elif idx < 6*n:
        # lower2: vec[idx] -> matrix[idx-5n+2, idx-5n]
        # vec[5n] -> (2,0), vec[5n+1] -> (3,1), ..., vec[6n-3] -> (n-1, n-3)
        # vec[6n-2:6n] are unused (constrained to 0)
        return (idx-5*n+2, idx-5*n) if idx-5*n+2 < n else None
    else:
        # lower3: vec[idx] -> matrix[idx-6n+3, idx-6n]
        # vec[6n] -> (3,0), vec[6n+1] -> (4,1), ..., vec[7n-4] -> (n-1, n-4)
        # vec[7n-3:7n] are unused (constrained to 0)
        return (idx-6*n+3, idx-6*n) if idx-6*n+3 < n else None


def mat_i_to_vec_i_p1_5(i, j, n):
    '''
    Convert matrix indices (i, j) to 7n vector index.
    Returns None if (i, j) is outside the banded structure.
    '''
    k = j - i
    return (3 - k) * n + i + k


'''
Initial transportation plan (for p=1.5)
Parameters:
  mu, nu: measures
  n: sample points
'''
def x_init_p1_5(mu, nu, n):
  x = np.zeros(7*n)  # 7n vector
  x_marg = np.zeros(n)
  y_marg = np.zeros(n)

  mask1 = (mu != 0)
  mask2 = (nu != 0)
  mask = mask1 & mask2

  # Compute main diagonal values (indices 3n to 4n-1 of the 7n vector)
  diag_vals = np.zeros(n)
  diag_vals[mask] = ((mu[mask] * nu[mask]) / (mu[mask]**0.5 + nu[mask]**0.5)**2) * 4
  
  x[3*n:4*n] = diag_vals  # Set main diagonal part
  
  x_marg[mask] = diag_vals[mask] / mu[mask]
  y_marg[mask] = diag_vals[mask] / nu[mask]

  return x, x_marg, y_marg, mask1, mask2


'''
Function to define the gradient of UOT (for p=1.5)
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  mask1, mask2: masks for the gradient
  n: dimension
'''
def grad_p1_5(x_marg, y_marg, mask1, mask2, n):
  # Compute derivatives only where masks are true
  dx = np.zeros(n)
  dy = np.zeros(n)
  dx[mask1] = dUp_dx(x_marg[mask1], 1.5)
  dy[mask2] = dUp_dx(y_marg[mask2], 1.5)
  
  # Initialize 7n gradient vector
  grad = np.zeros(7*n)
  
  # Main diagonal: grad[3n+i] corresponds to (i,i)
  # Cost: c[i,i] = 0, so grad = 0 + dx[i] + dy[i]
  m = mask1 & mask2
  grad[3*n:4*n][m] = dx[m] + dy[m]
  
  # Upper1 diagonal: grad[2n+i+1] corresponds to (i, i+1)
  # Cost: c[i,i+1] = 1
  mask_u1 = mask1[:-1] & mask2[1:]
  grad[2*n+1:3*n][mask_u1] = 1 + dx[:-1][mask_u1] + dy[1:][mask_u1]
  
  # Upper2 diagonal: grad[n+i+2] corresponds to (i, i+2)
  # Cost: c[i,i+2] = 2
  mask_u2 = mask1[:-2] & mask2[2:]
  grad[n+2:2*n][mask_u2] = 2 + dx[:-2][mask_u2] + dy[2:][mask_u2]
  
  # Upper3 diagonal: grad[i+3] corresponds to (i, i+3)
  # Cost: c[i,i+3] = 3
  mask_u3 = mask1[:-3] & mask2[3:]
  grad[3:n][mask_u3] = 3 + dx[:-3][mask_u3] + dy[3:][mask_u3]
  
  # Lower1 diagonal: grad[4n+i] corresponds to (i+1, i)
  # Cost: c[i+1,i] = 1
  mask_l1 = mask1[1:] & mask2[:-1]
  grad[4*n:5*n-1][mask_l1] = 1 + dx[1:][mask_l1] + dy[:-1][mask_l1]
  
  # Lower2 diagonal: grad[5n+i] corresponds to (i+2, i)
  # Cost: c[i+2,i] = 2
  mask_l2 = mask1[2:] & mask2[:-2]
  grad[5*n:6*n-2][mask_l2] = 2 + dx[2:][mask_l2] + dy[:-2][mask_l2]
  
  # Lower3 diagonal: grad[6n+i] corresponds to (i+3, i)
  # Cost: c[i+3,i] = 3
  mask_l3 = mask1[3:] & mask2[:-3]
  grad[6*n:7*n-3][mask_l3] = 3 + dx[3:][mask_l3] + dy[:-3][mask_l3]
  
  return grad


'''
Function to find the search direction
Parameters:
  pi: transportation plan
  grad: gradient of UOT
  M: upper bound for generalized simplex
  eps: tolerance
'''
def LMO_p1_5(pi, grad, M, eps):
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
def gap_calc_p1_5(grad, v, M, sum_term):
  if v[0] != -1:
      gap = - M * grad[v[0]] + sum_term
  else:
      gap = sum_term
  return gap


'''
Armijo stepsize
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  grad: gradient of UOT (7n vector)
  mu, nu: measures
  v: search direction (FW_idx_7n, AFW_idx_7n) where indices are in 7n vector space
  p: main parameter that defines the p-entropy
  coords: pre-computed matrix coordinates (FW_i, FW_j, AFW_i, AFW_j)
  theta, beta, gamma: parameters for the Armijo stepsize
'''
def armijo_p1_5(x_marg, y_marg, grad, mu, nu, v, p, coords, theta = 1, beta = 0.4, gamma = 0.5):
  # get the 7n indices and pre-computed matrix coordinates
  FW, AFW = v
  FW_i, FW_j, AFW_i, AFW_j = coords

  def cost_ij(ii, jj):
    return float(abs(ii - jj))

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
Stepsize calculation
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  grad: gradient of UOT (7n vector)
  mu, nu: measures
  v: search direction (FW_idx_7n, AFW_idx_7n)
  p: main parameter
  coords: pre-computed matrix coordinates (FW_i, FW_j, AFW_i, AFW_j)
  theta, beta, gamma: Armijo parameters
'''
def step_calc_p1_5(x_marg, y_marg, grad, mu, nu, v, p=1.5, 
                   coords=None, theta = 1, beta = 0.4, gamma = 0.5):
   return armijo_p1_5(x_marg, y_marg, grad, mu, nu, v, p, 
                      coords, theta = theta, beta = beta, gamma = gamma)


'''
Function to update the gradient of UOT (7n vector representation)
Parameters:
  x_marg, y_marg  : X and Y marginals of the transportation plan
  grad            : gradient of UOT (7n vector)
  mask1, mask2    : masks for the gradient (mu != 0 and nu != 0)
  coords          : pre-computed matrix coordinates (FW_i, FW_j, AFW_i, AFW_j)
'''
def update_grad_p1_5(x_marg, y_marg, grad, mask1, mask2, coords):
  n = x_marg.shape[0]
  p = 1.5  # Since this function is specific to p=1.5
  
  FW_i, FW_j, AFW_i, AFW_j = coords
  
  def update_row(i):
    '''Update gradient entries in row i (main and all diagonals)'''
    dUp_x = dUp_dx(x_marg[i], p)
    
    # (i, i) main diagonal at index 3n+i
    if mask2[i]:
      grad[3*n + i] = dUp_dx(y_marg[i], p) + dUp_x
    
    # (i, i+1) upper1 diagonal at index 2n+i+1
    if (i + 1 < n) and mask2[i + 1]:
      grad[2*n + i + 1] = 1 + dUp_dx(y_marg[i + 1], p) + dUp_x
    
    # (i, i+2) upper2 diagonal at index n+i+2
    if (i + 2 < n) and mask2[i + 2]:
      grad[n + i + 2] = 2 + dUp_dx(y_marg[i + 2], p) + dUp_x
    
    # (i, i+3) upper3 diagonal at index i+3
    if (i + 3 < n) and mask2[i + 3]:
      grad[i + 3] = 3 + dUp_dx(y_marg[i + 3], p) + dUp_x
    
    # (i, i-1) lower1 diagonal at index 4n+i-1
    if (i > 0) and mask2[i - 1]:
      grad[4*n + i - 1] = 1 + dUp_dx(y_marg[i - 1], p) + dUp_x
    
    # (i, i-2) lower2 diagonal at index 5n+i-2
    if (i > 1) and mask2[i - 2]:
      grad[5*n + i - 2] = 2 + dUp_dx(y_marg[i - 2], p) + dUp_x
    
    # (i, i-3) lower3 diagonal at index 6n+i-3
    if (i > 2) and mask2[i - 3]:
      grad[6*n + i - 3] = 3 + dUp_dx(y_marg[i - 3], p) + dUp_x
  
  def update_col(j):
    '''Update gradient entries in column j (main and all diagonals)'''
    dUp_y = dUp_dx(y_marg[j], p)
    
    # (j, j) main diagonal at index 3n+j
    if mask1[j]:
      grad[3*n + j] = dUp_y + dUp_dx(x_marg[j], p)
    
    # (j-1, j) upper1 diagonal at index 2n+j
    if (j > 0) and mask1[j - 1]:
      grad[2*n + j] = 1 + dUp_y + dUp_dx(x_marg[j - 1], p)
    
    # (j-2, j) upper2 diagonal at index n+j
    if (j > 1) and mask1[j - 2]:
      grad[n + j] = 2 + dUp_y + dUp_dx(x_marg[j - 2], p)
    
    # (j-3, j) upper3 diagonal at index j
    if (j > 2) and mask1[j - 3]:
      grad[j] = 3 + dUp_y + dUp_dx(x_marg[j - 3], p)
    
    # (j+1, j) lower1 diagonal at index 4n+j
    if (j + 1 < n) and mask1[j + 1]:
      grad[4*n + j] = 1 + dUp_y + dUp_dx(x_marg[j + 1], p)
    
    # (j+2, j) lower2 diagonal at index 5n+j
    if (j + 2 < n) and mask1[j + 2]:
      grad[5*n + j] = 2 + dUp_y + dUp_dx(x_marg[j + 2], p)
    
    # (j+3, j) lower3 diagonal at index 6n+j
    if (j + 3 < n) and mask1[j + 3]:
      grad[6*n + j] = 3 + dUp_y + dUp_dx(x_marg[j + 3], p)
  
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
  grad_xk: gradient vector (7n)
  xk: current transportation plan vector (7n)
  coords: pre-computed matrix coordinates (FW_i, FW_j, AFW_i, AFW_j)
  n: size parameter
  sign: +1 to add contributions, -1 to subtract contributions
'''
def update_sum_term_p1_5(sum_term, grad_xk, xk, coords, n, sign):
    FW_i, FW_j, AFW_i, AFW_j = coords
    coord_set = set()
    
    # Process each affected coordinate
    for i, j in [(FW_i, FW_j), (AFW_i, AFW_j)]:
        if i == -1:
            continue
        
        # Update all entries in row i and column j
        # Main diagonal (i,i)
        idx = mat_i_to_vec_i_p1_5(i, i, n)
        if idx is not None: coord_set.add(idx)
        
        # Upper diagonals from row i
        for k in range(1, 4):
            if i + k < n:
                idx = mat_i_to_vec_i_p1_5(i, i+k, n)
                if idx is not None: coord_set.add(idx)
        
        # Lower diagonals from row i
        for k in range(1, 4):
            if i - k >= 0:
                idx = mat_i_to_vec_i_p1_5(i, i-k, n)
                if idx is not None: coord_set.add(idx)
        
        # Column j: (i,j), (i-1,j), (i-2,j), (i-3,j), (i+1,j), (i+2,j), (i+3,j)
        # Main diagonal (j,j)
        idx = mat_i_to_vec_i_p1_5(j, j, n)
        if idx is not None: coord_set.add(idx)
        
        # Upper diagonals to column j (rows above j)
        for k in range(1, 4):
            if j - k >= 0:
                idx = mat_i_to_vec_i_p1_5(j-k, j, n)
                if idx is not None: coord_set.add(idx)
        
        # Lower diagonals to column j (rows below j)
        for k in range(1, 4):
            if j + k < n:
                idx = mat_i_to_vec_i_p1_5(j+k, j, n)
                if idx is not None: coord_set.add(idx)

    for idx in coord_set:
        sum_term += sign * grad_xk[idx] * xk[idx]

    return sum_term


'''
Apply step update for either AFW (Away Frank-Wolfe) or FW (Frank-Wolfe) direction
Parameters:
  xk, x_marg, y_marg: transportation plan (7n vector) and marginals (n vectors)
  grad_xk: gradient of UOT (7n vector)
  mu, nu: measures
  M: upper bound for generalized simplex
  vk: search direction indices (FW_idx_7n, AFW_idx_7n)
  coords: matrix coordinates (FW_i, FW_j, AFW_i, AFW_j)
Returns:
  updated xk, x_marg, y_marg
'''
def apply_step_p1_5(xk, x_marg, y_marg, grad_xk, mu, nu, M, vk, coords):
  FW, AFW = vk  # 7n vector indices
  FW_i, FW_j, AFW_i, AFW_j = coords  # matrix coordinates

  if AFW_i != -1:
    gamma0 = xk[AFW] - 1e-10
    gammak = step_calc_p1_5(x_marg, y_marg, grad_xk, mu, nu, vk, p=1.5,
                            coords=coords, theta=gamma0)
    xk[AFW] -= gammak
    x_marg[AFW_i] -= gammak / mu[AFW_i]
    y_marg[AFW_j] -= gammak / nu[AFW_j]
    if FW_i != -1:
      xk[FW] += gammak
      x_marg[FW_i] += gammak / mu[FW_i]
      y_marg[FW_j] += gammak / nu[FW_j]
  else:
    gamma0 = M - np.sum(xk) + xk[FW] - 1e-10
    gammak = step_calc_p1_5(x_marg, y_marg, grad_xk, mu, nu, vk, p=1.5,
                            coords=coords, theta=gamma0)
    xk[FW] += gammak
    x_marg[FW_i] += gammak / mu[FW_i]
    y_marg[FW_j] += gammak / nu[FW_j]

  return xk, x_marg, y_marg


'''
Pairwise Frank-Wolfe (specific for p = 1.5)
Parameters:
  mu, nu: measures
  M: upper bound for generalized simplex
  max_iter: max iterations
  delta, eps: tolerance
'''
def PW_FW_dim1_p1_5(mu, nu, M,
                    max_iter = 100, delta = 0.01, eps = 0.001):
  n = np.shape(mu)[0]

  # initial transportation plan, marginals and gradient initialization
  xk, x_marg, y_marg, mask1, mask2 = x_init_p1_5(mu, nu, n)
  grad_xk = grad_p1_5(x_marg, y_marg, mask1, mask2, n)
  
  # Initialize sum_term for efficient gap calculation
  sum_term = np.sum(grad_xk * xk)

  for k in range(max_iter):
    # search direction (returns 7n vector indices)
    vk = LMO_p1_5(xk, grad_xk, M, eps)

    # gap calculation
    gap = gap_calc_p1_5(grad_xk, vk, M, sum_term)

    if (gap <= delta) or (vk == (-1, -1)):
      print("FW_1dim_p1_5 converged after: ", k, " iterations ")
      return xk, grad_xk, x_marg, y_marg

    # Convert 7n indices to matrix coordinates (done once per iteration)
    FW, AFW = vk
    FW_i, FW_j = (-1, -1)
    AFW_i, AFW_j = (-1, -1)
    
    if FW != -1:
      result = vec_i_to_mat_i_p1_5(FW, n)
      if result is not None:
        FW_i, FW_j = result
    
    if AFW != -1:
      result = vec_i_to_mat_i_p1_5(AFW, n)
      if result is not None:
        AFW_i, AFW_j = result

    # Remove contributions from affected entries before gradient update
    sum_term = update_sum_term_p1_5(sum_term, grad_xk, xk, (FW_i, FW_j, AFW_i, AFW_j), n, sign=-1)

    # Apply step update
    xk, x_marg, y_marg = apply_step_p1_5(
        xk, x_marg, y_marg, grad_xk, mu, nu, M, vk,
        coords=(FW_i, FW_j, AFW_i, AFW_j)
    )
    
    # gradient update
    grad_xk = update_grad_p1_5(x_marg, y_marg, grad_xk, mask1, mask2, 
                               coords=(FW_i, FW_j, AFW_i, AFW_j))

    # Add back contributions from affected entries after gradient update
    sum_term = update_sum_term_p1_5(sum_term, grad_xk, xk, (FW_i, FW_j, AFW_i, AFW_j), n, sign=+1)

  print("FW_1dim_p1_5 converged after: ", max_iter, " iterations ")
  return xk, grad_xk, x_marg, y_marg
