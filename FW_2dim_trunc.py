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

    mask_x = (mu != 0)
    mask_y = (nu != 0)
    # Compute entropy only on non-zero measure indices
    term_x = np.sum(mu[mask_x] * Up(x_marg[mask_x] + s_i, p))
    term_y = np.sum(nu[mask_y] * Up(y_marg[mask_y] + s_j, p))

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
    
    # Sort by: 1) distance, 2) whether it's axis-aligned (|di|+|dj| for tie-break)
    # This puts axis-aligned neighbors before diagonals at same Euclidean distance
    displacements.sort(key=lambda x: (x[2], abs(x[0]) + abs(x[1])))
    
    # Extract just the costs in sorted order
    c = np.array([cost for _, _, cost in displacements])
    
    return c


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
    p: main parameter that defines the p-entropy
    n: dimension
    R: truncation radius
'''
def grad_trunc_dim2(x_marg, y_marg, mask1, mask2, c, p, n, R):
    grid_size = 2 * R - 1
    center = R - 1
    
    # Initialize gradient
    grad_x = np.zeros((grid_size**2, n, n))
    
    # Compute derivatives
    dx = np.zeros((n, n))
    dy = np.zeros((n, n))
    dx[mask1] = dUp_dx(x_marg[mask1], p)
    dy[mask2] = dUp_dx(y_marg[mask2], p)
    
    # Iterate through each displacement
    for k in range(grid_size**2):
        # Get 2D displacement
        grid_i = k // grid_size
        grid_j = k % grid_size
        di = grid_i - center
        dj = grid_j - center
        
        # Determine valid source and target slices
        if di >= 0:
            source_i_slice = slice(0, n - di)
            target_i_slice = slice(di, n)
        else:
            source_i_slice = slice(-di, n)
            target_i_slice = slice(0, n + di)
        
        if dj >= 0:
            source_j_slice = slice(0, n - dj)
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
    grad_si = np.where(mask1, 1/2*R + dx, 0)
    grad_sj = np.where(mask2, 1/2*R + dy, 0)
    
    return grad_x, (grad_si, grad_sj)


'''
Linear Minimization Oracle (LMO) for the transportation plan
Parameters:
    pi: current transportation plan
    grad_x: gradient with respect to the transport plan
    grad_s: gradient with respect to the truncated supports
    M: upper bound for generalized simplex
    eps: direction tolerance
'''
def LMO_x_dim2(pi, grad_x, M, eps):
    return