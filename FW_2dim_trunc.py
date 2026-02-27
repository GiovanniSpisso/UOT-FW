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
Compute an upper bound for the UOT cost:
Parameters:
  cost_trunc: truncated UOT cost
  n: number of samples
  si: truncated support on X
  R: truncation radius
  mu: measure on X
'''
def UOT_cost_upper_dim2(cost_trunc, n, si, R, mu):
  K = np.sqrt(2) * (n - 1) - R # Supposing c = np.sqrt(|x1-x2|^2 + |y1-y2|^2)
  return cost_trunc + K * np.sum(si * mu)


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
    grad_si = np.where(mask1, 1/2*R + dx, 0)
    grad_sj = np.where(mask2, 1/2*R + dy, 0)
    
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
    mu, nu: measures
    mask1, mask2: masks for valid entries in gradients
'''
def LMO_trunc_dim2_s(si, sj, grad_s, M, eps, mu, nu, mask1, mask2):
    grad_si, grad_sj = grad_s

    # Frank-Wolfe direction (minimize gradient)
    if (grad_si[mask1].min() + grad_sj[mask2].min()) < -eps:
        # Find 2D positions of minima
        FW_si = np.unravel_index(np.argmin(grad_si[mask1]), grad_si.shape)
        FW_sj = np.unravel_index(np.argmin(grad_sj[mask2]), grad_sj.shape)
    else:
        FW_si, FW_sj = (-1, -1), (-1, -1)
    
    # Away Frank-Wolfe direction (maximize gradient among active set)
    mask_si = (si > 0) & mask1
    mask_sj = (sj > 0) & mask2
    
    if not np.any(mask_si) and not np.any(mask_sj):
        return (FW_si, FW_sj, (-1, -1), (-1, -1))
    
    # Mask: only active entries with valid measure
    grad_si_masked = np.where(mask_si, grad_si, -np.inf)
    grad_sj_masked = np.where(mask_sj, grad_sj, -np.inf)

    max_val_si = grad_si_masked.max()
    max_val_sj = grad_sj_masked.max()

    if (max_val_si + max_val_sj) <= eps:
        if (np.sum(si*mu + sj*nu) < M):
            return (FW_si, FW_sj, (-1, -1), (-1, -1))
        else:
            print(f"M: {M}, sum(si*mu + sj*nu): {np.sum(si*mu + sj*nu):.2f}. Increase M!")
            return (FW_si, FW_sj, (-1, -1), (-1, -1))

    # Find 2D positions of maxima
    AFW_si = np.unravel_index(np.argmax(grad_si_masked), grad_si_masked.shape)
    AFW_sj = np.unravel_index(np.argmax(grad_sj_masked), grad_sj_masked.shape)

    return (FW_si, FW_sj, AFW_si, AFW_sj)


'''
Gap calculation for truncated problem
Parameters:
  x: current transportation plan
  grad_x: gradient with respect to the transport plan
  comp_FW: compact index of the FW direction for x (or -1 if no FW)
  M: upper bound for generalized simplex
  s_i, s_j: current truncated supports
  grad_s: gradient with respect to the truncated supports
  FW_v_s: LMO result for the truncated supports
  mu, nu: measures
'''
def gap_calc_trunc_dim2(x, grad_x, comp_FW, M, s_i, s_j, grad_s, FW_v_s, mu, nu):
    # ---- gap for supports s_i, s_j ----
    grad_si, grad_sj = grad_s
    gap_s = np.sum(s_i * mu * grad_si) + np.sum(s_j * nu * grad_sj)

    FW_si, FW_sj = FW_v_s
    if FW_si != (-1, -1):
        gap_s -= M * (grad_si[FW_si] + grad_sj[FW_sj])

    # ---- gap for transport plan xk ----
    gap_x = np.sum(x * grad_x)

    if comp_FW[0] != -1:
        gap_x -= M * grad_x[comp_FW]

    return gap_x + gap_s


'''
Armijo line search for 2D truncated UOT, sparse evaluation.
Parameters:
    x_marg, y_marg: X and Y marginals of the transportation plan
    grad_x: gradient wrt compact plan, shape ((2R-1)^2, n, n)
    grad_s: (grad_si, grad_sj), each shape (n, n)
    mu, nu: measures
    vk_x: ((comp_FW, full_FW), (comp_AFW, full_AFW))
        comp_* = (mat_idx, i, j) in compact representation
        full_* = (x1, x2, y1, y2) in full coordinates
    vk_s: (FW_si, FW_sj, AFW_si, AFW_sj) each either (a,b) or (-1,-1)
    s_i, s_j: truncated supports
    c_trunc: cost vector for truncated problem
    p: entropy parameter
    R: truncation radius
    theta, beta, gamma: Armijo parameters
'''
def armijo_trunc_dim2(x_marg, y_marg, grad_x, grad_s, mu, nu, vk_x, vk_s,
                      s_i, s_j, c_trunc, p, R,
                      theta=1.0, beta=0.4, gamma=0.5):

    (comp_FW, full_FW), (comp_AFW, full_AFW) = vk_x
    FW_si, FW_sj, AFW_si, AFW_sj = vk_s
    grad_si, grad_sj = grad_s

    # directional derivative <grad, d>
    inner = 0.0
    if comp_FW[0] != -1:
        inner += grad_x[comp_FW]
    if comp_AFW[0] != -1:
        inner -= grad_x[comp_AFW]
    if FW_si != (-1, -1):
        inner += grad_si[FW_si]
    if AFW_si != (-1, -1):
        inner -= grad_si[AFW_si]
    if FW_sj != (-1, -1):
        inner += grad_sj[FW_sj]
    if AFW_sj != (-1, -1):
        inner -= grad_sj[AFW_sj]

    # Linear transport-cost coefficient
    cost_lin = 0.0
    if comp_FW[0] != -1:
        cost_lin += c_trunc[comp_FW[0]]
    if comp_AFW[0] != -1:
        cost_lin -= c_trunc[comp_AFW[0]]

    # Linear penalty coefficient (matches your objective change for S_j)
    penalty_lin = 0.0
    if FW_sj != (-1, -1):
        penalty_lin += R
    if AFW_sj != (-1, -1):
        penalty_lin -= R

    # Precompute affected marginal entries and coefficients:
    # For each affected (i,j), net delta is coeff * theta, where coeff is ±1/mu or ±1/nu.
    dx_updates = {}  # (i,j) -> (a = x_marg+s_i at base, mu_ij, coeff)
    dy_updates = {}  # (k,l) -> (b = y_marg+s_j at base, nu_kl, coeff)

    def add_dx(key, coeff):
        i, j = key
        if i == -1:
            return
        a = x_marg[i, j] + s_i[i, j]
        if key in dx_updates:
            a0, mu0, coeff0 = dx_updates[key]
            dx_updates[key] = (a0, mu0, coeff0 + coeff)
        else:
            dx_updates[key] = (a, mu[i, j], coeff)

    def add_dy(key, coeff):
        k, l = key
        if k == -1:
            return
        b = y_marg[k, l] + s_j[k, l]
        if key in dy_updates:
            b0, nu0, coeff0 = dy_updates[key]
            dy_updates[key] = (b0, nu0, coeff0 + coeff)
        else:
            dy_updates[key] = (b, nu[k, l], coeff)

    # From plan FW/AFW (full indices)
    if full_FW[0] != -1:
        x1, x2, y1, y2 = full_FW
        add_dx((x1, x2), +1.0 / mu[x1, x2])
        add_dy((y1, y2), +1.0 / nu[y1, y2])

    if full_AFW[0] != -1:
        x1, x2, y1, y2 = full_AFW
        add_dx((x1, x2), -1.0 / mu[x1, x2])
        add_dy((y1, y2), -1.0 / nu[y1, y2])

    # From supports (normalized)
    if FW_si != (-1, -1):
        add_dx(FW_si, +1.0 / mu[FW_si])
    if AFW_si != (-1, -1):
        add_dx(AFW_si, -1.0 / mu[AFW_si])

    if FW_sj != (-1, -1):
        add_dy(FW_sj, +1.0 / nu[FW_sj])
    if AFW_sj != (-1, -1):
        add_dy(AFW_sj, -1.0 / nu[AFW_sj])

    def obj_change(theta_val):
        diff = theta_val * (cost_lin + penalty_lin)

        for _, (a, mu_ij, coeff) in dx_updates.items():
            d = coeff * theta_val
            diff += (Up(a + d, p) - Up(a, p)) * mu_ij

        for _, (b, nu_kl, coeff) in dy_updates.items():
            d = coeff * theta_val
            diff += (Up(b + d, p) - Up(b, p)) * nu_kl

        return diff

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
def step_calc_trunc_dim2(x_marg, y_marg, grad_x, grad_s, 
                         mu, nu, vk_x, vk_s, s_i, s_j, c_trunc, p, R,
                         theta=1.0, beta=0.4, gamma=0.5):
    return armijo_trunc_dim2(x_marg, y_marg, grad_x, grad_s, mu, nu, vk_x, vk_s, 
                             s_i, s_j, c_trunc, p, R, theta=theta, beta=beta, gamma=gamma)


'''
Compute maximum step size respecting all constraints on x, s_i, s_j
Parameters:
    x: transportation plan, shape ((2R-1)**2, n, n)
    s_i, s_j: truncated supports
    FW_x / AFW_x: indices for x, compact indices (mat_idx, i, j) or (-1,-1,-1)
    FW_si/AFW_si/FW_sj/AFW_sj: indices for s, (i,j) or -1
    M: upper bound for generalized simplex
    mu, nu: measures
'''
def compute_gamma_max_trunc_dim2(x, s_i, s_j, FW_x, AFW_x, FW_si, AFW_si, FW_sj, AFW_sj, M, mu, nu):
    gamma_max = np.inf

    # constraints from x
    if AFW_x is not None and AFW_x[0] != -1:
        gamma_max = min(gamma_max, x[AFW_x])
    elif FW_x is not None and FW_x[0] != -1:
        gamma_max = min(gamma_max, M - np.sum(x) + x[FW_x])

    # constraints from s_i
    if FW_si != (-1,-1):
        gamma_max = min(gamma_max, M - np.sum(s_i*mu) + s_i[FW_si]*mu[FW_si])
    if AFW_si != (-1,-1):
        gamma_max = min(gamma_max, s_i[AFW_si]*mu[AFW_si])

    # constraints from s_j
    if FW_sj != (-1,-1):
        gamma_max = min(gamma_max, M - np.sum(s_j*nu) + s_j[FW_sj]*nu[FW_sj])
    if AFW_sj != (-1,-1):
        gamma_max = min(gamma_max, s_j[AFW_sj]*nu[AFW_sj])

    return gamma_max


def build_disp_to_k(displacement_map):
    """Map (di,dj) -> k index in the compact representation."""
    return {d: k for k, d in enumerate(displacement_map)}

'''
Gradient update for 2D truncated representation.
Parameters
    x_marg, y_marg : (n,n) arrays
    s_i, s_j       : (n,n) arrays
    grad_x         : (K,n,n) array, K = (2R-1)^2
    grad_s         : tuple (grad_si, grad_sj), each (n,n)
    mask1, mask2   : (n,n) boolean masks
    c_trunc        : (K,) cost vector aligned with displacement_map
    displacement_map : list[(di,dj)] length K
    p, R           : parameters
    vk_x           : ((comp_FW, full_FW), (comp_AFW, full_AFW))
                     comp_* = (mat_idx,i,j) or (-1,-1,-1)
                     full_* = (x1,x2,y1,y2) or (-1,-1,-1,-1)
    vk_s           : (FW_si, FW_sj, AFW_si, AFW_sj) where each is (i,j) or (-1,-1)
'''
def update_grad_trunc_dim2(x_marg, y_marg, s_i, s_j, grad_x, grad_s,
                           mask1, mask2, c_trunc, displacement_map,
                           p, R, vk_x, vk_s):
    (_, full_FW), (_, full_AFW) = vk_x
    FW_si, FW_sj, AFW_si, AFW_sj = vk_s
    grad_si, grad_sj = grad_s

    n = x_marg.shape[0]
    K = (2 * R - 1) ** 2

    # Collect affected source/target pixels (in full grid coordinates)
    affected_src = set()
    affected_tgt = set()

    if full_FW[0] != -1:
        x1, x2, y1, y2 = full_FW
        affected_src.add((x1, x2))
        affected_tgt.add((y1, y2))
    if full_AFW[0] != -1:
        x1, x2, y1, y2 = full_AFW
        affected_src.add((x1, x2))
        affected_tgt.add((y1, y2))

    if FW_si != (-1, -1):
        affected_src.add(FW_si)
    if AFW_si != (-1, -1):
        affected_src.add(AFW_si)

    if FW_sj != (-1, -1):
        affected_tgt.add(FW_sj)
    if AFW_sj != (-1, -1):
        affected_tgt.add(AFW_sj)

    # Compute dUp_dx only where needed
    dx_cache = {}
    dy_cache = {}

    def get_dx(i, j):
        key = (i, j)
        if key not in dx_cache:
            dx_cache[key] = dUp_dx(x_marg[i, j] + s_i[i, j], p) if mask1[i, j] else 0.0
        return dx_cache[key]

    def get_dy(k, l):
        key = (k, l)
        if key not in dy_cache:
            dy_cache[key] = dUp_dx(y_marg[k, l] + s_j[k, l], p) if mask2[k, l] else 0.0
        return dy_cache[key]

    # Update grad_s at affected pixels
    for (i, j) in affected_src:
        if mask1[i, j]:
            grad_si[i, j] = 0.5 * R + get_dx(i, j)

    for (k, l) in affected_tgt:
        if mask2[k, l]:
            grad_sj[k, l] = 0.5 * R + get_dy(k, l)

    # Update grad_x entries impacted by affected sources
    # For fixed source (i,j): grad_x[k,i,j] changes for all k where target in bounds & mask2.
    for (i, j) in affected_src:
        dx_val = get_dx(i, j)

        for k_idx in range(K):
            di, dj = displacement_map[k_idx]
            ti = i + di
            tj = j + dj
            if 0 <= ti < n and 0 <= tj < n and mask2[ti, tj]:
                grad_x[k_idx, i, j] = c_trunc[k_idx] + dx_val + get_dy(ti, tj)

    # Update grad_x entries impacted by affected targets
    # For fixed target (k,l): all sources (i,j) such that (i+di,j+dj)=(k,l) for some displacement.
    # i.e. for each displacement (di,dj): source = (k-di, l-dj) updates entry grad_x[k_idx, source]
    for (k, l) in affected_tgt:
        dy_val = get_dy(k, l)

        for k_idx in range(K):
            di, dj = displacement_map[k_idx]
            i = k - di
            j = l - dj
            if 0 <= i < n and 0 <= j < n and mask1[i, j]:
                grad_x[k_idx, i, j] = c_trunc[k_idx] + get_dx(i, j) + dy_val

    return grad_x, (grad_si, grad_sj)


'''
Apply one pairwise FW step for the 2D truncated problem.
Parameters:
    are the natural 2D analogues of apply_step_trunc (1D), but:
  - vk_x = (i_FW, i_AFW) with i_* = (compact_*, full_*)
  - compact_* = (mat_idx, i, j) or (-1,-1,-1)
  - full_*    = (x1, x2, y1, y2) or (-1,-1,-1,-1)
  - vk_s = (FW_si, FW_sj, AFW_si, AFW_sj) where each is (i,j) or (-1,-1)
'''
def apply_step_trunc_dim2(xk, x_marg, y_marg, s_i, s_j, grad_xk_x, grad_xk_s,
                          mu, nu, M, vk_x, vk_s, c_trunc, p, R):
    (comp_FW, full_FW), (comp_AFW, full_AFW) = vk_x
    FW_si, FW_sj, AFW_si, AFW_sj = vk_s

    # Max feasible step
    gamma_max = compute_gamma_max_trunc_dim2(
        xk, s_i, s_j,
        FW_x=comp_FW, AFW_x=comp_AFW,
        FW_si=FW_si, AFW_si=AFW_si,
        FW_sj=FW_sj, AFW_sj=AFW_sj,
        M=M, mu=mu, nu=nu)

    # Armijo step with theta upper bound = gamma_max
    gammak = step_calc_trunc_dim2(
        x_marg, y_marg, grad_xk_x, grad_xk_s,
        mu, nu, vk_x, vk_s,
        s_i, s_j, c_trunc, p, R,
        theta=gamma_max)

    # Update plan xk and marginals via full indices
    # Away part
    if comp_AFW[0] != -1:
        xk[comp_AFW] -= gammak
        x1, x2, y1, y2 = full_AFW
        x_marg[x1, x2] -= gammak / mu[x1, x2]
        y_marg[y1, y2] -= gammak / nu[y1, y2]

        if comp_FW[0] != -1:
            xk[comp_FW] += gammak
            x1, x2, y1, y2 = full_FW
            x_marg[x1, x2] += gammak / mu[x1, x2]
            y_marg[y1, y2] += gammak / nu[y1, y2]

    # Pure FW part
    elif comp_FW[0] != -1:
        xk[comp_FW] += gammak
        x1, x2, y1, y2 = full_FW
        x_marg[x1, x2] += gammak / mu[x1, x2]
        y_marg[y1, y2] += gammak / nu[y1, y2]

    # Update supports s_i, s_j
    if FW_si != (-1, -1):
        s_i[FW_si] += gammak / mu[FW_si]
        s_j[FW_sj] += gammak / nu[FW_sj]
    if AFW_si != (-1, -1):
        s_i[AFW_si] -= gammak / mu[AFW_si]
        s_j[AFW_sj] -= gammak / nu[AFW_sj]

    return xk, x_marg, y_marg, s_i, s_j


'''
Pairwise Frank-Wolfe for 2D truncated UOT.
Parameters:
    mu, nu: measures (n x n arrays)
    M: upper bound for generalized simplex
    p: entropy parameter
    R: truncation radius
    max_iter: maximum iterations
    delta: convergence threshold for the gap
    eps: tolerance for LMO directions
'''
def PW_FW_dim2_trunc(mu, nu, M, p, R,
                     max_iter=100, delta=0.01, eps=0.001):
    n = mu.shape[0]

    # cost + displacement map
    c_trunc, displacement_map = cost_matrix_trunc_dim2(R)

    # initialization
    xk, x_marg, y_marg, mask1, mask2 = x_init_trunc_dim2(mu, nu, n, R, p)
    s_i = np.zeros((n, n))
    s_j = np.zeros((n, n))

    grad_xk_x, grad_xk_s = grad_trunc_dim2(x_marg, y_marg, mask1, mask2, c_trunc, displacement_map, p, n, R)

    for k in range(max_iter):
        # LMO
        i_FW, i_AFW = LMO_trunc_dim2_x(xk, grad_xk_x, displacement_map, M, eps=eps)
        vk_x = (i_FW, i_AFW)
        FW_si, FW_sj, AFW_si, AFW_sj = LMO_trunc_dim2_s(s_i, s_j, grad_xk_s, M, eps, mu, nu, mask1, mask2)
        vk_s = (FW_si, FW_sj, AFW_si, AFW_sj)

        # gap
        gap = gap_calc_trunc_dim2(xk, grad_xk_x, i_FW[0], M, s_i, s_j, grad_xk_s, (FW_si, FW_sj), mu, nu)

        # stopping
        no_x_dir = (i_FW[0][0] == -1 and i_AFW[0][0] == -1)
        no_s_dir = (FW_si == (-1, -1) and AFW_si == (-1, -1))
        if (gap <= delta) or (no_x_dir and no_s_dir):
            print("FW_2dim_trunc converged after:", k, "iterations")
            return xk, (grad_xk_x, grad_xk_s), x_marg, y_marg, s_i, s_j

        # step
        xk, x_marg, y_marg, s_i, s_j = apply_step_trunc_dim2(
            xk, x_marg, y_marg, s_i, s_j, grad_xk_x, grad_xk_s,
            mu, nu, M, vk_x, vk_s, c_trunc, p, R)

        # gradient update (incremental)
        grad_xk_x, grad_xk_s = update_grad_trunc_dim2(
            x_marg, y_marg, s_i, s_j, grad_xk_x, grad_xk_s,
            mask1, mask2, c_trunc, displacement_map, p, R, vk_x, vk_s)
        
    print("FW_2dim_trunc reached max iterations:", max_iter)
    return xk, (grad_xk_x, grad_xk_s), x_marg, y_marg, s_i, s_j