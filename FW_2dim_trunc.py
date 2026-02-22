import numpy as np

'''
Power-like entropy function
Parameters:
  x: transportation plan (assumed x > 0 or x == 0)
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
    elif p < 1:
        result = np.zeros_like(x, dtype=float)
        mask_nonzero = (x > 0)
        result[mask_nonzero] = (x[mask_nonzero]**(p-1) - 1) / (p - 1)
    else:
        result = (x**(p - 1) - 1) / (p - 1)
    
    return result


"""
Compute the truncated UOT cost:
Parameters:
  pi: transportation plan
  x_marg, y_marg: X and Y marginals of the transportation plan
  mu, nu: measures
"""
def truncated_cost_dim2(pi, x_marg, y_marg, c, mu, nu, p, s_i, s_j, R):
    # pi is a (2R-1)^2 * n^2 matrix: 
    # cost: array of length (2R-1)^2
    cost_transport = sum(c[k] * np.sum(pi[k]) for k in range(len(pi)))

    # Compute entropy only on non-zero measure indices
    term_x = np.sum(mu * Up(x_marg + s_i, p))
    term_y = np.sum(nu * Up(y_marg + s_j, p))

    C3 = R * np.sum(s_j * nu)

    return cost_transport + term_x + term_y + C3


'''
Compute the total UOT cost:
Parameters:
  pi: transportation plan
  x_marg, y_marg: X and Y marginals of the transportation plan
  c: cost function (vector form)
  mu, nu: measures
'''
def UOT_cost_upper_dim2(cost_trunc, n, si, R):
  K = np.sqrt(2) * (n - 1) - R # Supposing c = np.sqrt(|x1-x2|^2 + |y1-y2|^2)
  return cost_trunc + K * np.sum(si)


'''
Compute the cost matrix for 2D truncated optimal transport
Parameters:
R : truncation radius (the grid will be (2R-1) x (2R-1))
Returns:
c : np.ndarray
    Cost matrix of shape ((2R-1)^2,) in vectorized form
    Each entry is the Euclidean distance from the center (R-1, R-1)
displacement_map : list of tuples
    List of (di, dj) displacements corresponding to each cost entry
'''
def cost_matrix_trunc_dim2(R):
    grid_size = 2 * R - 1
    center = R - 1
    
    # Collect all (displacement, cost) pairs
    displacements = []
    
    for idx in range(grid_size**2):
        i = idx // grid_size
        j = idx % grid_size
        di = i - center
        dj = j - center
        cost = np.sqrt(di**2 + dj**2)
        displacements.append((di, dj, cost))
    
    # Sort by: 1) distance, 2) whether it's axis-aligned
    displacements.sort(key=lambda x: (x[2], abs(x[0]) + abs(x[1])))
    
    # Extract costs and displacement mapping
    c = np.array([cost for _, _, cost in displacements])
    displacement_map = [(di, dj) for di, dj, _ in displacements]
    
    return c, displacement_map


'''
Initial transportation plan + marginals
Parameters:
  mu, nu: measures
  n: sample points
  R: truncation radius
  p: entropy parameter
'''
def x_init_trunc_dim2(mu, nu, n, R, p):
    x = np.zeros(((2*R-1)**2, n, n)) # (2R-1)^2 matrices of shape n * n    
    x_marg = np.zeros((n, n))
    y_marg = np.zeros((n, n))

    mask1 = (mu != 0)
    mask2 = (nu != 0)
    mask = mask1 & mask2

    # Compute values only where mask is True, otherwise 0
    diag_vals = np.zeros((n, n))
    if np.any(mask):
        if p == 2:
            diag_vals[mask] = 2 * mu[mask] * nu[mask] / (mu[mask] + nu[mask])
        elif p == 1:
            diag_vals[mask] = np.sqrt(mu[mask] * nu[mask])
        elif p < 1:
            diag_vals[mask] = ((mu[mask]**(p-1) + nu[mask]**(p-1)) / (2 * (mu[mask]**(p-1) * nu[mask]**(p-1))))**(1/(1-p))
        elif p > 1:  
            diag_vals[mask] = ((mu[mask] * nu[mask]) / (mu[mask]**(p-1) + nu[mask]**(p-1))**(1/(p-1))) * 2**(1/(p-1))

    x[0] = diag_vals
    x_marg[mask] = diag_vals[mask] / mu[mask]
    y_marg[mask] = diag_vals[mask] / nu[mask]

    return x, x_marg, y_marg, mask1, mask2


'''
Function to define the gradient of UOT with respect to the transport plan 
and to truncated supports S_i, S_j in O(n^2)
Parameters:
    x_marg, y_marg: X and Y marginals of the transportation plan
    mask1, mask2: masks for the gradient
    c: cost vector for the truncated problem
    displacement_map: list of (di, dj) displacements corresponding to each cost entry
    p: main parameter that defines the p-entropy
    n: dimension
    R: truncation radius
'''
def grad_trunc_dim2(x_marg, y_marg, mask1, mask2, c, displacement_map, p, n, R):
    grid_size = 2 * R - 1
    
    # Initialize gradient
    grad_x = np.zeros((grid_size**2, n, n))
    
    # Compute derivatives
    dx = np.zeros((n, n))
    dy = np.zeros((n, n))
    dx[mask1] = dUp_dx(x_marg[mask1], p)
    dy[mask2] = dUp_dx(y_marg[mask2], p)
    
    # Iterate through each band k using the displacement map
    for k, (di, dj) in enumerate(displacement_map):
        # Determine valid source and target slices
        if di >= 0:
            source_i_slice = slice(0, n - di) if di > 0 else slice(0, n)
            target_i_slice = slice(di, n)
        else:
            source_i_slice = slice(-di, n)
            target_i_slice = slice(0, n + di)
        
        if dj >= 0:
            source_j_slice = slice(0, n - dj) if dj > 0 else slice(0, n)
            target_j_slice = slice(dj, n)
        else:
            source_j_slice = slice(-dj, n)
            target_j_slice = slice(0, n + dj)
        
        # Extract relevant slices
        dx_source = dx[source_i_slice, source_j_slice]
        dy_target = dy[target_i_slice, target_j_slice]
        mask_source = mask1[source_i_slice, source_j_slice]
        mask_target = mask2[target_i_slice, target_j_slice]
        
        # Combined mask
        mask = mask_source & mask_target
        
        # Compute gradient
        grad_x[k, source_i_slice, source_j_slice][mask] = (
            c[k] + dx_source[mask] + dy_target[mask]
        )
    
    # Gradients for truncated supports
    grad_si = np.where(mask1, R + dx, 0)
    grad_sj = np.where(mask2, R + dy, 0)
    
    return grad_x, (grad_si, grad_sj)


'''
Linear Minimization Oracle for truncated 2D representation.
Parameters:
  pi: transportation plan, shape ((2R-1)^2, n, n)
  grad: gradient of UOT, shape ((2R-1)^2, n, n)
  displacement_map: list of (di, dj) tuples for each matrix index
  M: upper bound for generalized simplex
  eps: tolerance
Returns:
  i_FW: Tuple ((mat_idx, i, j), (i, j, k, l)) or ((-1,-1,-1), (-1,-1,-1,-1))
  i_AFW: Tuple ((mat_idx, i, j), (i, j, k, l)) or ((-1,-1,-1), (-1,-1,-1,-1))
'''
def LMO_trunc_dim2_x(pi, grad, displacement_map, M, eps=0.001):
    n = pi.shape[1]
    
    def compact_to_full(matrix_idx, i, j):
        """Convert compact (mat_idx, i, j) to full (i, j, k, l)."""
        if matrix_idx == -1:
            return (-1, -1, -1, -1)
        di, dj = displacement_map[matrix_idx]
        k = i + di
        l = j + dj
        return (i, j, k, l)
    
    # Frank-Wolfe direction (minimize gradient)
    flat_idx = np.argmin(grad)
    min_val = grad.flat[flat_idx]
    
    if min_val < -eps:
        # Manual unraveling: flat_idx -> (matrix_idx, i, j)
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
Linear Minimization Oracle for truncated supports S_i, S_j.
Parameters:
    si, sj: truncated supports
    grad_s: tuple of gradients (grad_si, grad_sj)
    M: upper bound for generalized simplex
    eps: tolerance
    mask1, mask2: masks for valid positions in si and sj
'''
def LMO_trunc_dim2_s(si, sj, grad_s, M, eps, mask1, mask2):
    grad_si, grad_sj = grad_s
    
    # Mask invalid positions (set to inf to exclude from argmin)
    grad_si_valid = np.where(mask1, grad_si, np.inf)
    grad_sj_valid = np.where(mask2, grad_sj, np.inf)

    # Frank-Wolfe direction (minimize gradient)
    if (grad_si_valid.min() + grad_sj_valid.min()) < -eps:
        # Find 2D positions of minima
        FW_si = np.unravel_index(np.argmin(grad_si_valid), grad_si_valid.shape)
        FW_sj = np.unravel_index(np.argmin(grad_sj_valid), grad_sj_valid.shape)
    else:
        FW_si, FW_sj = -1, -1
    
    # Away Frank-Wolfe direction (maximize gradient among active set)
    mask_si = (si > 0)
    mask_sj = (sj > 0)
    
    if not np.any(mask_si) and not np.any(mask_sj):
        return (FW_si, FW_sj, -1, -1)
    
    # Mask: only active entries with valid measure
    grad_si_masked = np.where(mask_si & mask1, grad_si, -np.inf)
    grad_sj_masked = np.where(mask_sj & mask2, grad_sj, -np.inf)

    max_val_si = grad_si_masked.max()
    max_val_sj = grad_sj_masked.max()

    if (max_val_si + max_val_sj) <= eps:
        if (np.sum(si + sj) < M):
            return (FW_si, FW_sj, -1, -1)
        else:
            print(f"M: {M}, sum(si+sj): {np.sum(si+sj):.2f}. Increase M!")
            return (FW_si, FW_sj, -1, -1)

    # Find 2D positions of maxima
    AFW_si = np.unravel_index(np.argmax(grad_si_masked), grad_si_masked.shape)
    AFW_sj = np.unravel_index(np.argmax(grad_sj_masked), grad_sj_masked.shape)

    return (FW_si, FW_sj, AFW_si, AFW_sj)


'''
Gap calculation for truncated problem
'''
def gap_calc_trunc_dim2():
    pass