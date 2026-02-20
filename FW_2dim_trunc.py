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
def precompute_cost_matrix_dim2(R):
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


