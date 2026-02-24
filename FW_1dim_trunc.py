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
    

'''
Compute the truncated UOT cost:
Parameters:
  pi: transportation plan
  x_marg, y_marg: X and Y marginals of the transportation plan
  c: cost function (vector form)
  mu, nu: measures
  p: main parameter that defines the p-entropy
  s_i, s_j: indices of truncated support
  R: truncation radius
'''
def truncated_cost(pi, x_marg, y_marg, c, mu, nu, p, s_i, s_j, R):
  C1 = np.multiply(c, pi).sum()
  
  cost_row = np.sum(mu * Up(x_marg + s_i, p))
  cost_col = np.sum(nu * Up(y_marg + s_j, p))
  C2 = cost_row + cost_col

  C3 = R * np.sum(s_j * nu)

  return C1 + C2 + C3


'''
Compute the total UOT cost:
Parameters:
  pi: transportation plan
  x_marg, y_marg: X and Y marginals of the transportation plan
  c: cost function (vector form)
  mu, nu: measures
'''
def UOT_cost_upper(cost_trunc, n, si, R, mu):
  K = n - 1 - R # Supposing c = |i-j|
  return cost_trunc + K * np.sum(si * mu)


def vector_to_matrix(vec, n, R):
    '''
    Convert a vector representation of a banded matrix back to full matrix form.
    
    Parameters:
      vec: 1D array representing the banded matrix
      n: dimension of the matrix (n x n)
      R: truncation radius (diagonals from -R to R)
    
    Returns:
      matrix: Full n x n matrix
    '''
    matrix = np.zeros((n, n))
    pos = 0
    
    for k in range(-R + 1, R):
        m = n - abs(k)
        
        if k >= 0:
            i = np.arange(m)
            j = i + k
        else:
            j = np.arange(m)
            i = j - k
        
        # Place the diagonal elements
        matrix[i, j] = vec[pos:pos + m]
        pos += m
    
    return matrix


def vector_index_to_matrix_indices(idx, n, R):
    """
    Convert vector index to matrix indices (i, j) directly.
    
    Parameters:
      idx: index in the vector representation
      n: dimension of the matrix
      R: truncation radius
    
    Returns:
      (i, j): matrix indices
    """
    # Find which diagonal k and position within that diagonal
    pos = 0
    for k in range(-R + 1, R):
        m = n - abs(k)
        if idx < pos + m:
            offset = idx - pos
            if k >= 0:
                return (offset, offset + k)
            else:
                return (offset - k, offset)
        pos += m
    raise ValueError(f"Index {idx} out of bounds")


def matrix_indices_to_vector_index(i, j, n, R):
    """
    Convert matrix indices (i, j) to vector index.
    Returns None if (i, j) is outside the truncated support.
    
    Parameters:
      i, j: matrix indices
      n: dimension of the matrix
      R: truncation radius
    
    Returns:
      idx: index in the vector representation, or None if outside support
    """
    k = j - i
    
    # Check if diagonal k is within truncation radius
    if not (-R < k < R):
        return None
    
    # Find position of diagonal k
    pos = 0
    for diag in range(-R + 1, k):
        pos += n - abs(diag)
    
    # Find position within diagonal k
    if k >= 0:
        offset = i
    else:
        offset = j
    
    return pos + offset


'''
Initial transportation plan + marginals
Parameters:
    mu, nu: measures
    n: number of samples
    c: cost function
    p: main parameter that defines the p-entropy
'''
def x_init_trunc(mu, nu, n, c, p):
    x = np.zeros_like(c, dtype=float)
    mask_c = (c == 0)         
    x_marg = np.zeros(n)
    y_marg = np.zeros(n)

    mask1 = (mu != 0)
    mask2 = (nu != 0)
    mask = mask1 & mask2

    # Compute values only where mask is True, otherwise 0
    diag_values = np.zeros_like(mu, dtype=float)
    if np.any(mask):
        if p == 2:
            diag_values[mask] = 2 * mu[mask] * nu[mask] / (mu[mask] + nu[mask])
        elif p == 1:
            diag_values[mask] = np.sqrt(mu[mask] * nu[mask])
        elif p < 1:
            diag_values[mask] = ((mu[mask]**(p-1) + nu[mask]**(p-1)) / (2 * (mu[mask]**(p-1) * nu[mask]**(p-1))))**(1/(1-p))
        elif p > 1:  
            diag_values[mask] = ((mu[mask] * nu[mask]) / (mu[mask]**(p-1) + nu[mask]**(p-1))**(1/(p-1))) * 2**(1/(p-1))
    
    x[mask_c] = diag_values

    x_diag = x[mask_c]
    x_marg[mask] = x_diag[mask] / mu[mask]
    y_marg[mask] = x_diag[mask] / nu[mask]

    return x, x_marg, y_marg, mask1, mask2

'''
Function to define the gradient of UOT with respect to the transport plan 
and to truncated supports S_i, S_j in O(n)
Parameters:
    x_marg, y_marg: X and Y marginals of the transportation plan
    mask1, mask2: masks for the gradient
    c: cost function
    p: main parameter that defines the p-entropy
    n: dimension
    R: truncation radius
'''
def grad_trunc(x_marg, y_marg, mask1, mask2, c, p, n, R):
    grad_x = np.zeros_like(c, dtype=float)
    
    # Compute derivatives only where mask is true to avoid log(0)
    dx = np.zeros_like(x_marg)
    dy = np.zeros_like(y_marg)
    dx[mask1] = dUp_dx(x_marg[mask1], p)
    dy[mask2] = dUp_dx(y_marg[mask2], p)
    
    pos = 0
    for k in range(-R + 1, R):
        m = n - abs(k)

        if k >= 0:
            i = np.arange(m)
            j = i + k
        else:
            j = np.arange(m)
            i = j - k

        mask = mask1[i] & mask2[j]
        grad_x[pos:pos + m][mask] = abs(k) + dx[i][mask] + dy[j][mask]
        pos += m
    
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
def LMO_x(pi, grad_x, M, eps):
    # Frank-Wolfe direction
    idx = np.argmin(grad_x)
    min_val = grad_x[idx]
    if min_val < -eps:
        FW_x = idx
    else:
        FW_x = -1
        
    # Away Frank-Wolfe direction
    mask = (pi > 0)
    
    if not np.any(mask):
        return (FW_x, -1)
    else:
        grad_masked = np.where(mask, grad_x, -np.inf)

        max_val = grad_masked.max()
        if (max_val <= eps):
            if (pi.sum() < M):
                return (FW_x, -1)
            else:
                print("M: ", M, ", pi.sum(): ", pi.sum(), ". Increase M!")

        AFW_x = np.argmax(grad_masked)
        
    return (FW_x, AFW_x)


'''
Linear Minimization Oracle (LMO) for the truncated supports S_i, S_j
Parameters:
    si, sj: current truncated supports
    grad_s: gradient with respect to the truncated supports
    M: upper bound for generalized simplex
    eps: direction tolerance
    mu, nu: measures 
'''
def LMO_s(si, sj, grad_s, M, eps, mu, nu):

    if (grad_s[0].min() + grad_s[1].min()) < -eps:
        FW_si = int(np.argmin(grad_s[0]))
        FW_sj = int(np.argmin(grad_s[1]))
    else:
        FW_si, FW_sj = -1, -1
    
    mask_si = (si > 0)
    mask_sj = (sj > 0)
    if not np.any(mask_si) and not np.any(mask_sj):
        return (FW_si, FW_sj, -1, -1)
    else:
        grad_si_masked = np.where(mask_si, grad_s[0], -np.inf)
        grad_sj_masked = np.where(mask_sj, grad_s[1], -np.inf)

        max_val_si = grad_si_masked.max()
        max_val_sj = grad_sj_masked.max()

        if (max_val_si + max_val_sj) <= eps:
            if (np.sum(si*mu + sj*nu) < M):
                return (FW_si, FW_sj, -1, -1)
            else:
                print("M: ", M, ", sum(si*mu + sj*nu): ", np.sum(si*mu + sj*nu), ". Increase M!")

        AFW_si = np.argmax(grad_si_masked)
        AFW_sj = np.argmax(grad_sj_masked)

        return (FW_si, FW_sj, AFW_si, AFW_sj)
    

'''
Parameters:
    xk: current transportation plan
    grad_x: gradient with respect to the transport plan
    vk_x: LMO result for the transportation plan
    M: upper bound for generalized simplex
    s_i, s_j: current truncated supports
    grad_s: gradient with respect to the truncated supports
    vk_s: LMO result for the truncated supports
    mu, nu: measures
'''
def gap_calc_trunc(xk, grad_x, vk_x, M, s_i, s_j, grad_s, vk_s, mu, nu):
    gap_s = np.dot(s_i*mu, grad_s[0]) + np.dot(s_j*nu, grad_s[1])
    if vk_s[0] != -1:
        gap_s -= M * (grad_s[0][vk_s[0]] + grad_s[1][vk_s[1]])
    
    gap_x = np.dot(xk, grad_x)
    if vk_x[0] != -1:
        gap_x -= M * grad_x[vk_x[0]]
    
    return gap_x + gap_s


'''
Armijo stepsize for truncated UOT
Parameters:
    x_marg, y_marg: X and Y marginals of the transportation plan
    grad_x: gradient of UOT with respect to x
    grad_s: tuple (grad_si, grad_sj) - gradient with respect to s
    mu, nu: measures
    v_coords: (i_FW, j_FW, i_AFW, j_AFW) - matrix indices for x
    vk_x: (FW_x, AFW_x) - vector indices for x
    vk_s: (FW_si, FW_sj, AFW_si, AFW_sj) - indices for s
    s_i, s_j: current truncated supports
    c: cost function (vector form)
    p: main parameter
    R: truncation radius
    theta, beta, gamma: Armijo parameters
'''
def armijo(x_marg, y_marg, grad_x, grad_s, mu, nu, v_coords, vk_x, vk_s, 
           s_i, s_j, c, p, R, theta = 1.0, beta = 0.4, gamma = 0.5):
    
    i_FW, j_FW, i_AFW, j_AFW = v_coords
    FW_x, AFW_x = vk_x
    FW_si, FW_sj, AFW_si, AFW_sj = vk_s
    grad_si, grad_sj = grad_s
    
    # Directional derivative (inner product): what is the gradient at the FW/AFW directions
    inner = 0
    if FW_x != -1:
        inner += grad_x[FW_x]
    if AFW_x != -1:
        inner -= grad_x[AFW_x]
    if FW_si != -1:
        inner += grad_si[FW_si]
    if AFW_si != -1:
        inner -= grad_si[AFW_si]
    if FW_sj != -1:
        inner += grad_sj[FW_sj]
    if AFW_sj != -1:
        inner -= grad_sj[AFW_sj]
    
    # Objective change: sum of cost and entropy terms
    def obj_change(theta_val):
        diff = 0

        # Cost terms for x
        if FW_x != -1:
            diff += theta_val * c[FW_x]
        if AFW_x != -1:
            diff -= theta_val * c[AFW_x]

        # Net changes on marginals (avoid double counting when indices overlap)
        dx_marg = np.zeros_like(x_marg)
        dy_marg = np.zeros_like(y_marg)

        if i_FW != -1 and mu[i_FW] != 0:
            dx_marg[i_FW] += theta_val / mu[i_FW]
        if j_FW != -1 and nu[j_FW] != 0:
            dy_marg[j_FW] += theta_val / nu[j_FW]

        if i_AFW != -1 and mu[i_AFW] != 0:
            dx_marg[i_AFW] -= theta_val / mu[i_AFW]
        if j_AFW != -1 and nu[j_AFW] != 0:
            dy_marg[j_AFW] -= theta_val / nu[j_AFW]

        if FW_si != -1 and mu[FW_si] != 0:
            dx_marg[FW_si] += theta_val / mu[FW_si]
        if AFW_si != -1 and mu[AFW_si] != 0:
            dx_marg[AFW_si] -= theta_val / mu[AFW_si]

        if FW_sj != -1 and nu[FW_sj] != 0:
            dy_marg[FW_sj] += theta_val / nu[FW_sj]
        if AFW_sj != -1 and nu[AFW_sj] != 0:
            dy_marg[AFW_sj] -= theta_val / nu[AFW_sj]

        # Entropy terms for x_marg, y_marg (apply net changes per index)
        idx_x = np.nonzero(dx_marg)[0]
        for i in idx_x:
            diff += (Up(x_marg[i] + s_i[i] + dx_marg[i], p) - Up(x_marg[i] + s_i[i], p)) * mu[i]

        idx_y = np.nonzero(dy_marg)[0]
        for j in idx_y:
            diff += (Up(y_marg[j] + s_j[j] + dy_marg[j], p) - Up(y_marg[j] + s_j[j], p)) * nu[j]

        # R penalty term for s_j 
        penalty = 0
        if FW_sj != -1:
            penalty += 1
        if AFW_sj != -1:
            penalty -= 1
        diff += R * theta_val * penalty

        return diff
    
    # Armijo line search
    diff = obj_change(theta)
    while diff > beta * theta * inner:
        theta *= gamma
        diff = obj_change(theta)
    
    return theta


'''
Stepsize calculation
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  grad_x, grad_s: gradients of UOT
  mu, nu: measures
  vk_x, vk_s: search directions
  s_i, s_j: truncated supports
  c: cost function
  p: main parameter
  n, R: problem dimensions
  theta, beta, gamma: Armijo parameters
'''
def step_calc(x_marg, y_marg, grad_x, grad_s, 
              mu, nu, vk_x, vk_s, s_i, s_j, c, p, n, R, 
              theta = 1.0, beta = 0.4, gamma = 0.5):
    FW_ix, FW_jx, AFW_ix, AFW_jx = -1, -1, -1, -1
    if vk_x[0] != -1:
        FW_ix, FW_jx = vector_index_to_matrix_indices(vk_x[0], n=n, R=R)
    if vk_x[1] != -1:
        AFW_ix, AFW_jx = vector_index_to_matrix_indices(vk_x[1], n=n, R=R)
    v_coords = (FW_ix, FW_jx, AFW_ix, AFW_jx)

    step = armijo(x_marg, y_marg, grad_x, grad_s, mu, nu, v_coords, vk_x, vk_s, 
                  s_i, s_j, c, p, R, theta = theta, beta = beta, gamma = gamma)
    return step, FW_ix, FW_jx, AFW_ix, AFW_jx


'''
Compute maximum step size respecting all constraints on x, s_i, s_j
Parameters:
    xk: current transportation plan
    s_i, s_j: current truncated supports
    FW_x, AFW_x: vector indices for x search directions
    FW_si, AFW_si, FW_sj, AFW_sj: indices for s search directions
    M: upper bound for generalized simplex
    mu, nu: measures
'''
def compute_gamma_max(xk, s_i, s_j, FW_x, AFW_x, FW_si, AFW_si, FW_sj, AFW_sj, M, mu, nu):
    gamma_max = np.inf
    
    # Constraints from x coordinates
    if AFW_x != -1:
        gamma_max = min(gamma_max, xk[AFW_x])
    elif FW_x != -1:
        gamma_max = min(gamma_max, M - np.sum(xk) + xk[FW_x])
    
    # Constraints from s_i
    if FW_si != -1:
        gamma_max = min(gamma_max, M - np.sum(s_i*mu) + s_i[FW_si]*mu[FW_si])
    if AFW_si != -1:
        gamma_max = min(gamma_max, s_i[AFW_si])
    
    # Constraints from s_j
    if FW_sj != -1:
        gamma_max = min(gamma_max, M - np.sum(s_j*nu) + s_j[FW_sj]*nu[FW_sj])
    if AFW_sj != -1:
        gamma_max = min(gamma_max, s_j[AFW_sj])
    
    return gamma_max


'''
Function to update the gradient of plan + truncated supports
Only updates affected entries based on the vertices used (FW and AFW)
Parameters:
    x_marg, y_marg: X and Y marginals
    si, sj: truncated supports
    grad_x: gradient with respect to plan (vector form)
    grad_s: tuple (grad_si, grad_sj)
    mask1, mask2: masks for valid indices
    p: entropy parameter
    n: dimension
    R: truncation radius
    v_coords: (i_FW, j_FW, i_AFW, j_AFW) - matrix indices for x
    vk_s: LMO result for s (FW_si, FW_sj, AFW_si, AFW_sj) - matrix indices
'''
def update_grad_trunc(x_marg, y_marg, si, sj, grad_x, grad_s, 
                     mask1, mask2, p, n, R, v_coords, vk_s):
    i_FW, j_FW, i_AFW, j_AFW = v_coords
    FW_si, FW_sj, AFW_si, AFW_sj = vk_s
    
    # Collect affected indices
    affected_i = set()
    affected_j = set()
    
    # Add matrix indices from x updates
    if i_FW != -1:
        affected_i.add(i_FW)
        affected_j.add(j_FW)
    if i_AFW != -1:
        affected_i.add(i_AFW)
        affected_j.add(j_AFW)
    
    # Add indices from s updates
    if FW_si != -1:
        affected_i.add(FW_si)
    if AFW_si != -1:
        affected_i.add(AFW_si)
    if FW_sj != -1:
        affected_j.add(FW_sj)
    if AFW_sj != -1:
        affected_j.add(AFW_sj)
    
    # Compute updated derivative values only for affected indices
    grad_si, grad_sj = grad_s
    
    for i in affected_i:
        dx_i = dUp_dx(x_marg[i] + si[i], p)
        # Update grad_si
        grad_si[i] = 1/2*R + dx_i
        
        # Update grad_x for all diagonals containing row i
        for k in range(-R + 1, R):
            # Check if column j = i + k is valid
            j = i + k
            if 0 <= j < n and mask2[j]:
                # Find vector index for (i, j)
                idx = matrix_indices_to_vector_index(i, j, n, R)
                if idx is not None:
                    dy_j = dUp_dx(y_marg[j] + sj[j], p)
                    grad_x[idx] = abs(k) + dx_i + dy_j
    
    for j in affected_j:
        dy_j = dUp_dx(y_marg[j] + sj[j], p)
        # Update grad_sj
        grad_sj[j] = 1/2*R + dy_j
        
        # Update grad_x for all diagonals containing column j
        for k in range(-R + 1, R):
            # Check if row i = j - k is valid
            i = j - k
            if 0 <= i < n and mask1[i] and i not in affected_i:  # Skip if already updated
                # Find vector index for (i, j)
                idx = matrix_indices_to_vector_index(i, j, n, R)
                if idx is not None:
                    dx_i = dUp_dx(x_marg[i] + si[i], p)
                    grad_x[idx] = abs(k) + dx_i + dy_j
    
    return grad_x, (grad_si, grad_sj)


'''
Apply step update for truncated UOT
Parameters:
  xk: current transportation plan
  x_marg, y_marg: X and Y marginals
  s_i, s_j: truncated supports
  grad_xk_x: gradient with respect to plan
  grad_xk_s: gradient with respect to supports
  mu, nu: measures
  M: upper bound for generalized simplex
  vk_x: search directions for x
  vk_s: search directions for s
  c: cost function
  p: main parameter
  n, R: problem dimensions
'''
def apply_step_trunc(xk, x_marg, y_marg, s_i, s_j, grad_xk_x, grad_xk_s,
                     mu, nu, M, vk_x, vk_s, c, p, n, R):
    FW_x, AFW_x = vk_x
    FW_si, FW_sj, AFW_si, AFW_sj = vk_s
    
    # Compute maximum allowed step size respecting all constraints
    gamma_max = compute_gamma_max(xk, s_i, s_j, FW_x, AFW_x, FW_si, AFW_si, FW_sj, AFW_sj, M, mu, nu)
    
    # Compute step size using Armijo with gamma_max as upper bound
    result = step_calc(x_marg, y_marg, grad_xk_x, grad_xk_s,
                      mu, nu, vk_x, vk_s, s_i, s_j, c, p, n, R, 
                      theta = gamma_max)
    
    if isinstance(result, tuple):
        gammak, i_FW, j_FW, i_AFW, j_AFW = result
    else:
        gammak = result
        i_FW, j_FW, i_AFW, j_AFW = -1, -1, -1, -1

    # Update x coordinates
    if AFW_x != -1:
        xk[AFW_x] -= gammak
        x_marg[i_AFW] -= gammak / mu[i_AFW]
        y_marg[j_AFW] -= gammak / nu[j_AFW]
        
        if FW_x != -1:
            xk[FW_x] += gammak
            x_marg[i_FW] += gammak / mu[i_FW]
            y_marg[j_FW] += gammak / nu[j_FW]
    elif FW_x != -1:
        xk[FW_x] += gammak
        x_marg[i_FW] += gammak / mu[i_FW]
        y_marg[j_FW] += gammak / nu[j_FW]

    # Update s_i, s_j coordinates
    if FW_si != -1:
        s_i[FW_si] += gammak / mu[FW_si]
        s_j[FW_sj] += gammak / nu[FW_sj]
    if AFW_si != -1:
        s_i[AFW_si] -= gammak / mu[AFW_si]
        s_j[AFW_sj] -= gammak / nu[AFW_sj]
    
    return xk, x_marg, y_marg, s_i, s_j, (i_FW, j_FW, i_AFW, j_AFW)


'''
Pair-Wise Frank-Wolfe algorithm for 1D truncated UOT
Parameters:
    mu, nu: measures
    M: upper bound for generalized simplex
    p: main parameter that defines the p-entropy
    c: cost function
    R: truncation radius
    max_iter: maximum number of iterations
    delta: convergence tolerance
    eps: direction tolerance
'''
def PW_FW_dim1_trunc(mu, nu, M, p, c, R,
                     max_iter = 100, delta = 0.01, eps = 0.001):
    n = np.shape(mu)[0]

    # transportation plan, marginals, cost and gradient initialization
    xk, x_marg, y_marg, mask1, mask2 = x_init_trunc(mu, nu, n, c, p)

    s_i, s_j = np.zeros(n), np.zeros(n)
    grad_xk_x, grad_xk_s = grad_trunc(x_marg, y_marg, mask1, mask2, c, p, n, R)

    for k in range(max_iter):
        # LMO call
        vk_x = LMO_x(xk, grad_xk_x, M, eps)
        vk_s = LMO_s(s_i, s_j, grad_xk_s, M, eps, mu, nu)

        # gap calculation
        gap = gap_calc_trunc(xk, grad_xk_x, vk_x, M, s_i, s_j, grad_xk_s, vk_s, mu, nu)

        if (gap <= delta) or (vk_x == (-1, -1) and vk_s == (-1, -1, -1, -1)):
            print("Converged after: ", k, " iterations ")
            return xk, (grad_xk_x, grad_xk_s), x_marg, y_marg, s_i, s_j
        
        # Apply step update
        xk, x_marg, y_marg, s_i, s_j, v_coords = apply_step_trunc(
            xk, x_marg, y_marg, s_i, s_j, grad_xk_x, grad_xk_s,
            mu, nu, M, vk_x, vk_s, c, p, n, R)

        # Update gradient
        grad_xk_x, grad_xk_s = update_grad_trunc(x_marg, y_marg, s_i, s_j, grad_xk_x, grad_xk_s, 
                                                 mask1, mask2, p, n, R, v_coords, vk_s)

    return xk, (grad_xk_x, grad_xk_s), x_marg, y_marg, s_i, s_j