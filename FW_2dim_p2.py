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
  mu, nu: measures
"""
def cost_dim2_p2(pi, x_marg, y_marg, mu, nu):
    # pi is a 9 * n^2 matrix: 
    # [(i,j)|(i-1,j)|(i,j-1)|(i+1,j)|(i,j+1)|(i-1,j-1)|(i+1,j-1)|(i-1,j+1)|(i+1,j+1)]
    # cost: [0, 1, 1, 1, 1, sqrt(2), sqrt(2), sqrt(2), sqrt(2)]
    cost_transport = np.sum(pi[1] + pi[2] + pi[3] + pi[4]) + np.sqrt(2) * np.sum(pi[5] + pi[6] + pi[7] + pi[8])

    mask_x = (mu != 0)
    mask_y = (nu != 0)
    # Compute entropy only on non-zero measure indices
    term_x = np.sum(mu[mask_x] * Up(x_marg[mask_x], 2))
    term_y = np.sum(nu[mask_y] * Up(y_marg[mask_y], 2))

    return cost_transport + term_x + term_y


"""
Cost lookup using array indexing, valid for p = 2 and the specific 9-point stencil
Parameters:
  mat_ind: index from 0 to 8 corresponding to the relative position in the stencil
"""
cost_array = np.array([0, 1, 1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)])

def cost_ind_dim2_p2(mat_ind):
    return cost_array[mat_ind]


'''
Function for converting (9,n,n) matrix representation into its dense form (n,n,n,n)
Parameters:
  pi: transportation plan in (9,n,n) format
  n: number of sample points
'''
def to_dense_dim2_p2(pi, n):
    A = np.zeros((n, n, n, n))
    
    # Create index grids
    i_grid, j_grid = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    
    offsets = [
        (0, 0), (-1, 0), (0, -1), (1, 0), (0, 1),
        (-1, -1), (1, -1), (-1, 1), (1, 1)
    ]
    
    for idx, (di, dj) in enumerate(offsets):
        # Compute target indices
        k_grid = i_grid + di
        l_grid = j_grid + dj
        
        # Create validity mask
        valid = (k_grid >= 0) & (k_grid < n) & (l_grid >= 0) & (l_grid < n)
        
        # Extract valid indices
        i_valid = i_grid[valid]
        j_valid = j_grid[valid]
        k_valid = k_grid[valid]
        l_valid = l_grid[valid]
        
        # Assign values
        A[i_valid, j_valid, k_valid, l_valid] = pi[idx, i_valid, j_valid]
    
    return A


'''
Initial transportation plan + marginals (only for p = 2)
Parameters:
  mu, nu: measures
  n: sample points
'''
def x_init_dim2_p2(mu, nu, n):
  n = len(mu)
  x0 = np.zeros((9, n, n)) # 9 matrices of shape n * n
  x_marg = np.zeros((n,n))
  y_marg = np.zeros((n,n))

  mask1 = (mu != 0)
  mask2 = (nu != 0)
  mask = mask1 & mask2

  # Compute main diagonal values (indices in the first n^2 entries)
  diag_vals = np.zeros((n,n))
  diag_vals[mask] = 2 * mu[mask] * nu[mask] / (mu[mask] + nu[mask])

  x0[0] = diag_vals
  
  x_marg[mask] = diag_vals[mask] / mu[mask]
  y_marg[mask] = diag_vals[mask] / nu[mask]
  
  return x0, x_marg, y_marg, mask1, mask2


'''
Function to define the gradient of UOT (only for p = 2)
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  mask1, mask2: masks for the zero coordinates
  n: sample points
'''
def grad_dim2_p2(x_marg, y_marg, mask1, mask2, n):
    # Compute derivatives only where masks are true
    dx = np.zeros((n, n))
    dy = np.zeros((n, n))
    dx[mask1] = dUp_dx(x_marg[mask1], 2)
    dy[mask2] = dUp_dx(y_marg[mask2], 2)

    # Initialize 9 * n^2 gradient vector
    grad = np.zeros((9, n, n))

    # grad[mat_idx, i, j] = cost + dx[i,j] + dy[k,l]
    # where (k,l) = (i,j) + offset[mat_idx]

    # Index 0
    m = mask1 & mask2
    grad[0][m] = dx[m] + dy[m]  # cost = 0

    # Index 1: (i,j) → (i-1,j) [up]
    # Source: (i,j) with i ∈ [1,n-1]
    # Target: (i-1,j) = (k,j) with k ∈ [0,n-2]
    m = mask1[1:, :] & mask2[:-1, :]
    grad[1][1:, :][m] = 1 + dx[1:, :][m] + dy[:-1, :][m]

    # Index 2: (i,j) → (i,j-1) [left]
    m = mask1[:, 1:] & mask2[:, :-1]
    grad[2][:, 1:][m] = 1 + dx[:, 1:][m] + dy[:, :-1][m]

    # Index 3: (i,j) → (i+1,j) [down]
    m = mask1[:-1, :] & mask2[1:, :]
    grad[3][:-1, :][m] = 1 + dx[:-1, :][m] + dy[1:, :][m]

    # Index 4: (i,j) → (i,j+1) [right]
    m = mask1[:, :-1] & mask2[:, 1:]
    grad[4][:, :-1][m] = 1 + dx[:, :-1][m] + dy[:, 1:][m]

    # Diagonal directions
    sqrt2 = np.sqrt(2)
    
    # Index 5: (i,j) → (i-1,j-1) [up-left]
    m = mask1[1:, 1:] & mask2[:-1, :-1]
    grad[5][1:, 1:][m] = sqrt2 + dx[1:, 1:][m] + dy[:-1, :-1][m]

    # Index 6: (i,j) → (i+1,j-1) [down-left]
    m = mask1[:-1, 1:] & mask2[1:, :-1]
    grad[6][:-1, 1:][m] = sqrt2 + dx[:-1, 1:][m] + dy[1:, :-1][m]

    # Index 7: (i,j) → (i-1,j+1) [up-right]
    m = mask1[1:, :-1] & mask2[:-1, 1:]
    grad[7][1:, :-1][m] = sqrt2 + dx[1:, :-1][m] + dy[:-1, 1:][m]

    # Index 8: (i,j) → (i+1,j+1) [down-right]
    m = mask1[:-1, :-1] & mask2[1:, 1:]
    grad[8][:-1, :-1][m] = sqrt2 + dx[:-1, :-1][m] + dy[1:, 1:][m]

    return grad


'''
Linear Minimization Oracle for compact (9, n, n) representation.
Parameters:
  pi: transportation plan
  grad: gradient of UOT
  M: upper bound for generalized simplex
  eps: tolerance
Returns:
  i_FW: Tuple ((mat_idx, i, j), (i, j, k, l)) or ((-1,-1,-1), (-1,-1,-1,-1))
  i_AFW: Tuple ((mat_idx, i, j), (i, j, k, l)) or ((-1,-1,-1), (-1,-1,-1,-1))
'''
# Offset mapping for converting mat_idx to (di, dj)
offsets_dim2_p2 = [
    (0, 0),    # 0: same position
    (-1, 0),   # 1: up
    (0, -1),   # 2: left
    (1, 0),    # 3: down
    (0, 1),    # 4: right
    (-1, -1),  # 5: up-left
    (1, -1),   # 6: down-left
    (-1, 1),   # 7: up-right
    (1, 1),    # 8: down-right
]
offset_to_idx_dim2_p2 = {offset: idx for idx, offset in enumerate(offsets_dim2_p2)}

def LMO_dim2_p2(pi, grad, M, eps=0.001):
    
    n = pi.shape[1]
    
    def compact_to_full(matrix_idx, i, j):
        """Convert compact (mat_idx, i, j) to full (i, j, k, l)."""
        if matrix_idx == -1:
            return (-1, -1, -1, -1)
        di, dj = offsets_dim2_p2[matrix_idx]
        k = i + di
        l = j + dj
        return (i, j, k, l)
    
    # Frank-Wolfe direction (minimize gradient)
    flat_idx = np.argmin(grad)
    min_val = grad.flat[flat_idx]
    
    if min_val < -eps:
        # Manual unraveling
        matrix_idx = flat_idx // (n * n)
        position = flat_idx % (n * n)
        i = position // n
        j = position % n
        compact_FW = (matrix_idx, i, j)
        full_FW = compact_to_full(matrix_idx, i, j)
        i_FW = (compact_FW, full_FW)
    else:
        i_FW = ((-1, -1, -1), (-1, -1, -1, -1))

    # Away Frank-Wolfe direction (maximize gradient among active set)
    mask = (pi > 0)

    if not np.any(mask):
        return i_FW, ((-1, -1, -1), (-1, -1, -1, -1))
    
    grad_masked = np.where(mask, grad, -np.inf)
    max_val = grad_masked.max()
    
    if max_val <= eps:
        if pi.sum() < M:
            return i_FW, ((-1, -1, -1), (-1, -1, -1, -1))
        else:
            print(f"Warning: M={M}, pi.sum()={pi.sum():.2f}. Increase M!")
            return i_FW, ((-1, -1, -1), (-1, -1, -1, -1))

    flat_idx = np.argmax(grad_masked)
    
    # Manual unraveling
    matrix_idx = flat_idx // (n * n)
    position = flat_idx % (n * n)
    i = position // n
    j = position % n
    compact_AFW = (matrix_idx, i, j)
    full_AFW = compact_to_full(matrix_idx, i, j)
    i_AFW = (compact_AFW, full_AFW)
    
    return i_FW, i_AFW


'''
Parameters:
  grad: gradient of UOT
  compact_FW: indices of the FW direction
  M: upper bound for generalized simplex
  sum_term: sum of the transportation plan (for Frank-Wolfe gap calculation)
'''
def gap_calc_dim2_p2(grad, compact_FW, M, sum_term):
  if compact_FW[0] != -1:
    gap = - M * grad[compact_FW] + sum_term
  else:
    gap = sum_term

  return gap


'''
Optimal stepsize (p = 2)
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  mu, nu: measures
  mat_idx: matrix index of the FW and AFW vertices
  full: indices of the FW and AFW vertices in full (i, j, k, l) format
'''
def opt_step_dim2_p2(x_marg, y_marg, mu, nu, mat_idx, full):
  ind_FW, ind_AFW = mat_idx
  (x1FW, x2FW, y1FW, y2FW), (x1AFW, x2AFW, y1AFW, y2AFW) = full
  
  if x1FW == -1:
    return (cost_ind_dim2_p2(ind_AFW) + y_marg[y1AFW, y2AFW] +
            x_marg[x1AFW, x2AFW] - 2) / (1/mu[x1AFW, x2AFW] + 1/nu[y1AFW, y2AFW])
  elif x1AFW == -1:
    return (2 - cost_ind_dim2_p2(ind_FW) - y_marg[y1FW, y2FW] -
            x_marg[x1FW, x2FW]) / (1/mu[x1FW, x2FW] + 1/nu[y1FW, y2FW])
  elif x1FW == x1AFW and x2FW == x2AFW:
    return (cost_ind_dim2_p2(ind_AFW) - cost_ind_dim2_p2(ind_FW) + y_marg[y1AFW, y2AFW] -
            y_marg[y1FW, y2FW]) / (1/nu[y1AFW, y2AFW] + 1/nu[y1FW, y2FW])
  elif y1FW == y1AFW and y2FW == y2AFW:
    return (cost_ind_dim2_p2(ind_AFW) - cost_ind_dim2_p2(ind_FW) + x_marg[x1AFW, x2AFW] -
            x_marg[x1FW, x2FW]) / (1/mu[x1AFW, x2AFW] + 1/mu[x1FW, x2FW])
  else:
    return (cost_ind_dim2_p2(ind_AFW) - cost_ind_dim2_p2(ind_FW) +
            y_marg[y1AFW, y2AFW] - y_marg[y1FW, y2FW] + x_marg[x1AFW, x2AFW] -
            x_marg[x1FW, x2FW]) / (1/mu[x1AFW, x2AFW] + 1/mu[x1FW, x2FW] + 1/nu[y1AFW, y2AFW] + 1/nu[y1FW, y2FW])


'''
Gradient update: only compute derivatives for affected positions
Parameters:
    x_marg: X marginals (n, n)
    y_marg: Y marginals (n, n)
    grad: Gradient in compact form (9, n, n)
    mask1: Source mask (n, n)
    mask2: Target mask (n, n)
    FW_full: (x1FW, x2FW, y1FW, y2FW) or (-1, -1, -1, -1)
    AFW_full: (x1AFW, x2AFW, y1AFW, y2AFW) or (-1, -1, -1, -1)
'''
def grad_update_dim2_p2(x_marg, y_marg, grad, mask1, mask2, FW_full, AFW_full):
    n = grad.shape[1]
    
    def update_source_neighborhood(x1, x2, y1, y2):
        """Update entries with source near (x1,x2), target at (y1,y2)."""
        if not mask2[y1, y2]:
            return
        
        # Compute derivative only once for target
        dy_val = dUp_dx(y_marg[y1, y2], 2)
        
        # Iterate over 3×3 neighborhood of source
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                i, j = x1 + di, x2 + dj
                
                if not (0 <= i < n and 0 <= j < n and mask1[i, j]):
                    continue
                
                offset = (y1 - i, y2 - j)
                if offset in offset_to_idx_dim2_p2:
                    mat_idx = offset_to_idx_dim2_p2[offset]
                    # Compute derivative only for this specific (i, j)
                    dx_val = dUp_dx(x_marg[i, j], 2)
                    grad[mat_idx, i, j] = cost_ind_dim2_p2(mat_idx) + dx_val + dy_val
    
    def update_target_neighborhood(x1, x2, y1, y2):
        """Update entries with source at (x1,x2), target near (y1,y2)."""
        if not mask1[x1, x2]:
            return
        
        # Compute derivative only once for source
        dx_val = dUp_dx(x_marg[x1, x2], 2)
        
        # Iterate over 3×3 neighborhood of target
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                k, l = y1 + di, y2 + dj
                
                if not (0 <= k < n and 0 <= l < n and mask2[k, l]):
                    continue
                
                offset = (k - x1, l - x2)
                if offset in offset_to_idx_dim2_p2:
                    mat_idx = offset_to_idx_dim2_p2[offset]
                    # Compute derivative only for this specific (k, l)
                    dy_val = dUp_dx(y_marg[k, l], 2)
                    grad[mat_idx, x1, x2] = cost_ind_dim2_p2(mat_idx) + dx_val + dy_val
    
    # Update FW direction
    if FW_full[0] != -1:
        x1, x2, y1, y2 = FW_full
        update_source_neighborhood(x1, x2, y1, y2)
        update_target_neighborhood(x1, x2, y1, y2)
    
    # Update AFW direction
    if AFW_full[0] != -1:
        x1, x2, y1, y2 = AFW_full
        update_source_neighborhood(x1, x2, y1, y2)
        update_target_neighborhood(x1, x2, y1, y2)
    
    return grad


'''
Function to update the sum_term used in the gap calculation
Parameters:
    sum_term: current value of the sum term
    grad: gradient in compact form (9, n, n)
    x: transportation plan in compact form (9, n, n)
    FW_full, AFW_full: full indices (x1, x2, y1, y2) or (-1, -1, -1, -1)
    n: number of sample points
    sign: +1, -1
'''
def update_sum_term_dim2_p2(sum_term, grad, x, FW_full, AFW_full, n, sign):
    n = grad.shape[1]
    
    # Use a set to track processed entries (avoid double-counting)
    processed = set()
    
    def process_position(x1, x2, y1, y2):
        nonlocal sum_term
        
        if x1 == -1:
            return
        
        # Source neighborhood
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                i, j = x1 + di, x2 + dj
                if not (0 <= i < n and 0 <= j < n):
                    continue
                
                offset = (y1 - i, y2 - j)
                if offset in offset_to_idx_dim2_p2:
                    mat_idx = offset_to_idx_dim2_p2[offset]
                    entry = (mat_idx, i, j)
                    if entry not in processed:
                        sum_term += sign * grad[mat_idx, i, j] * x[mat_idx, i, j]
                        processed.add(entry)
        
        # Target neighborhood
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                k, l = y1 + di, y2 + dj
                if not (0 <= k < n and 0 <= l < n):
                    continue
                
                offset = (k - x1, l - x2)
                if offset in offset_to_idx_dim2_p2:
                    mat_idx = offset_to_idx_dim2_p2[offset]
                    entry = (mat_idx, x1, x2)
                    if entry not in processed:
                        sum_term += sign * grad[mat_idx, x1, x2] * x[mat_idx, x1, x2]
                        processed.add(entry)
    
    # Process FW and AFW
    if FW_full[0] != -1:
        process_position(*FW_full)
    if AFW_full[0] != -1:
        process_position(*AFW_full)
    
    return sum_term


"""
Apply step update for either AFW (Away Frank-Wolfe) or FW (Frank-Wolfe) direction
Parameters:
    xk: Transportation plan (9, n, n)
    x_marg, y_marg: Marginals (n, n)
    mu, nu: Measures (n, n)
    M: Upper bound
    FW_compact: (mat_FW, i_FW, j_FW)
    FW_full: (x1FW, x2FW, y1FW, y2FW)
    AFW_compact: (mat_AFW, i_AFW, j_AFW)
    AFW_full: (x1AFW, x2AFW, y1AFW, y2AFW)

Returns:
    Updated xk, x_marg, y_marg
"""
def apply_step_dim2_p2(xk, x_marg, y_marg, mu, nu, M, 
                       FW_compact, FW_full, AFW_compact, AFW_full):

    mat_FW, i_FW, j_FW = FW_compact
    mat_AFW, i_AFW, j_AFW = AFW_compact
    x1FW, x2FW, y1FW, y2FW = FW_full
    x1AFW, x2AFW, y1AFW, y2AFW = AFW_full
    
    # Compute optimal step size
    gammak_opt = opt_step_dim2_p2(x_marg, y_marg, mu, nu, 
                                  mat_idx = (mat_FW, mat_AFW), full = (FW_full, AFW_full))
    
    if x1AFW != -1:
        # AFW exists
        gamma0 = xk[mat_AFW, i_AFW, j_AFW] - 1e-10
        gammak = min(gammak_opt, gamma0)
        
        xk[mat_AFW, i_AFW, j_AFW] -= gammak
        x_marg[x1AFW, x2AFW] -= gammak / mu[x1AFW, x2AFW]
        y_marg[y1AFW, y2AFW] -= gammak / nu[y1AFW, y2AFW]
        
        if x1FW != -1:
            xk[mat_FW, i_FW, j_FW] += gammak
            x_marg[x1FW, x2FW] += gammak / mu[x1FW, x2FW]
            y_marg[y1FW, y2FW] += gammak / nu[y1FW, y2FW]
    
    else:
        # Only FW
        gamma0 = M - np.sum(xk) + xk[mat_FW, i_FW, j_FW] - 1e-10
        gammak = min(gammak_opt, gamma0)
        
        xk[mat_FW, i_FW, j_FW] += gammak
        x_marg[x1FW, x2FW] += gammak / mu[x1FW, x2FW]
        y_marg[y1FW, y2FW] += gammak / nu[y1FW, y2FW]
    
    return xk, x_marg, y_marg


'''
Pairwise Frank-Wolfe (only for p = 2)
Parameters:
  mu, nu: measures
  M: upper bound for generalized simplex
  step: stepsize calculation method
  max_iter: max iterations
  delta, eps: tolerance
'''
def PW_FW_dim2_p2(mu, nu, M,
                  max_iter = 100, delta = 0.01, eps = 0.001):
  n = np.shape(mu)[0]

  # transportation plan, marginals and gradient initialization
  xk, x_marg, y_marg, mask1, mask2 = x_init_dim2_p2(mu, nu, n)
  grad_xk = grad_dim2_p2(x_marg, y_marg, mask1, mask2, n)

  # Initialize sum_term for efficient gap calculation
  sum_term = np.sum(grad_xk * xk)

  for k in range(max_iter):
    # search direction (returns both (9,n,n) and (n,n,n,n) vector indices)
    (comp_FW, full_FW), (comp_AFW, full_AFW) = LMO_dim2_p2(xk, grad_xk, M, eps)

    # gap calculation
    gap = gap_calc_dim2_p2(grad_xk, comp_FW, M, sum_term)

    if (gap <= delta) or (full_FW == (-1,-1,-1,-1) and full_AFW == (-1,-1,-1,-1)): 
      print("Converged after: ", k, " iterations ")
      return (xk, grad_xk, x_marg, y_marg)

    # Remove contributions from affected coordinates before gradient update
    sum_term = update_sum_term_dim2_p2(sum_term, grad_xk, xk, full_FW, full_AFW, n, sign=-1)
    
    # Apply step update
    xk, x_marg, y_marg = apply_step_dim2_p2(xk, x_marg, y_marg, mu, nu, M, 
                                            comp_FW, full_FW, comp_AFW, full_AFW)

    # gradient update
    grad_xk = grad_update_dim2_p2(x_marg, y_marg, grad_xk, mask1, mask2, full_FW, full_AFW)
    
    # Add back contributions from affected coordinates after gradient update
    sum_term = update_sum_term_dim2_p2(sum_term, grad_xk, xk, full_FW, full_AFW, n, sign=1)

  print("Converged after: ", max_iter, " iterations ")
  return (xk, grad_xk, x_marg, y_marg)