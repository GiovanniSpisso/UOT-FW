import numpy as np

class TriDiagonal2D:
    OFFSETS = {
        ( 0,  0): "dist0",

        (-1,  0): "dist1_0",
        ( 0, -1): "dist1_1",
        ( 1,  0): "dist1_2",
        ( 0,  1): "dist1_3",

        (-1, -1): "dist2_0",
        ( 1, -1): "dist2_1",
        (-1,  1): "dist2_2",
        ( 1,  1): "dist2_3",
    }

    def __init__(self, **blocks):
        self.dist0   = blocks["dist0"]
        self.dist1_0 = blocks["dist1_0"]
        self.dist1_1 = blocks["dist1_1"]
        self.dist1_2 = blocks["dist1_2"]
        self.dist1_3 = blocks["dist1_3"]
        self.dist2_0 = blocks["dist2_0"]
        self.dist2_1 = blocks["dist2_1"]
        self.dist2_2 = blocks["dist2_2"]
        self.dist2_3 = blocks["dist2_3"]

        self.n = self.dist0.shape[0]

        self.blocks = {
            (0, 0): self.dist0,
            (-1, 0): self.dist1_0,
            (0, -1): self.dist1_1,
            (1, 0): self.dist1_2,
            (0, 1): self.dist1_3,
            (-1, -1): self.dist2_0,
            (1, -1): self.dist2_1,
            (-1, 1): self.dist2_2,
            (1, 1): self.dist2_3,
        }

    def get(self, i, j, k, l):
        di, dj = k - i, l - j
        block = self.blocks.get((di, dj))
        if block is None:
            return 0.0
        return block[i, j]

    def set(self, i, j, k, l, val):
        di, dj = k - i, l - j
        block = self.blocks.get((di, dj))
        if block is None:
            raise ValueError("Outside stencil")
        block[i, j] = val

    def update(self, i, j, k, l, delta):
        if delta == 0:
            return
        di, dj = k - i, l - j
        block = self.blocks.get((di, dj))
        if block is None:
            raise ValueError("Outside stencil")
        block[i, j] += delta

    def sum(self):
        return sum(b.sum() for b in self.blocks.values())

    def has_positive(self):
        return any(np.any(b > 0) for b in self.blocks.values())

    def min(self):
        return min(b.min() for b in self.blocks.values())

    def argmin(self):
        min_val = np.inf
        min_pos = (-1, -1, -1, -1)

        for (di, dj), block in self.blocks.items():
            val = block.min()
            if val < min_val:
                idx = np.unravel_index(np.argmin(block), block.shape)
                i, j = idx
                min_val = val
                min_pos = (i, j, i + di, j + dj)

        return min_pos

    def masked_argmax(self, mask, eps):
        best_val = eps
        best_pos = (-1, -1, -1, -1)

        for (di, dj), block in self.blocks.items():
            active = mask.blocks[(di, dj)] > 0
            if not np.any(active):
                continue

            masked_vals = np.where(active, block, -np.inf)
            val = masked_vals.max()
            if val > best_val:
                idx = np.unravel_index(np.argmax(masked_vals), block.shape)
                i, j = idx
                best_val = val
                best_pos = (i, j, i + di, j + dj)

        return best_val, best_pos

    def dot(self, other):
        return sum(np.sum(self.blocks[k] * other.blocks[k]) for k in self.blocks)

    def round(self, decimals=0):
        return TriDiagonal2D(**{
            name: np.round(getattr(self, name), decimals)
            for name in self.__dict__ if name.startswith("dist")
        })

    def to_dense(self):
        n = self.n
        A = np.zeros((n, n, n, n))
        for (di, dj), block in self.blocks.items():
            for i in range(n):
                for j in range(n):
                    k, l = i + di, j + dj
                    if 0 <= k < n and 0 <= l < n:
                        A[i, j, k, l] = block[i, j]
        return A
    

'''
Power-like entropy function
Parameters:
  x: transportation plan
  p: main parameter that defines the p-entropy
'''
def Up(x, p):
    x = np.asarray(x)

    neg = x < 0
    if np.any(neg & (x < -1e-14)):
        print("Attention! x < 0")

    x = np.maximum(x, 0)

    if p == 1:
        return x * np.log(x) - x + 1
    elif p == 0:
        return x - 1 - np.log(x)
    else:
        return (x**p - p * (x - 1) - 1) / (p * (p - 1))


'''
Derivative of power-like entropy function
Parameters:
  x: transportation plan
  p: main parameter that defines the p-entropy
'''
def dUp_dx(x, p):
    x = np.asarray(x)

    # clamp negatives to zero
    neg = x < 0
    if np.any(neg & (x < -1e-14)):
        print("Attention! x < 0 (gradient)")

    x = np.maximum(x, 0)

    if p == 1:
        return np.log(x)
    else:
        return (x**(p - 1) - 1) / (p - 1)


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
def UOT_cost_p2(pi, x_marg, y_marg, mu, nu, p):
    def transport_cost(pi):
      total = 0.0
      n = pi.n

      I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")

      for (di, dj), block in pi.blocks.items():
        K = I + di
        L = J + dj

        valid = (0 <= K) & (K < n) & (0 <= L) & (L < n)

        total += np.sum(
            cost(I[valid], J[valid], K[valid], L[valid]) *
            block[valid])

      return total

    def marginal_penalty(x_marg, y_marg, mu, nu, p):
      term_x = np.sum(mu * Up(x_marg, p))
      term_y = np.sum(nu * Up(y_marg, p))
      return term_x + term_y

    cost_transport = transport_cost(pi)
    cost_marginals = marginal_penalty(x_marg, y_marg, mu, nu, p)
    return cost_transport + cost_marginals


'''
Initial transportation plan + marginals (only for p = 2)
Parameters:
  mu, nu: measures
'''
def x_init_p2(mu, nu):
  n = len(mu)
  den = mu + nu
  x0 = np.zeros((n,n), dtype=float)
  x_marg = np.zeros((n,n), dtype=float)
  y_marg = np.zeros((n,n), dtype=float)

  mask1 = (mu != 0)
  mask2 = (nu != 0)

  x_marg[mask1] = 2 * nu[mask1] / den[mask1]
  y_marg[mask2] = 2 * mu[mask2] / den[mask2]

  x0[mask1] = 2 * mu[mask1] * nu[mask1] / den[mask1]
  x0[mask2] = 2 * mu[mask2] * nu[mask2] / den[mask2]
  x = TriDiagonal2D(
      dist0 = x0,
      dist1_0 = np.zeros((n, n)),
      dist1_1 = np.zeros((n, n)),
      dist1_2 = np.zeros((n, n)),
      dist1_3 = np.zeros((n, n)),
      dist2_0 = np.zeros((n, n)),
      dist2_1 = np.zeros((n, n)),
      dist2_2 = np.zeros((n, n)),
      dist2_3 = np.zeros((n, n))
  )
  return x, x_marg, y_marg, mask1, mask2


'''
Function to define the gradient of UOT with respect to one coordinate (only for p = 2)
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  mask1, mask2: masks for the zero coordinates
  n: dimension
'''
def grad_init_p2(x_marg, y_marg, mask1, mask2, n):
    # precompute marginals
    gX = np.zeros((n, n))
    gY = np.zeros((n, n))

    gY[mask1] = dUp_dx(x_marg[mask1], 2)  # p = 2
    gX[mask2] = dUp_dx(y_marg[mask2], 2)  # p = 2

    # initialize blocks
    dist0 = np.zeros((n,n))    # (i,j)

    dist1_0 = np.zeros((n,n))  # (i-1,j)
    dist1_1 = np.zeros((n,n))  # (i,j-1)
    dist1_2 = np.zeros((n,n))  # (i+1,j)
    dist1_3 = np.zeros((n,n))  # (i,j+1)

    dist2_0 = np.zeros((n,n))  # (i-1,j-1)
    dist2_1 = np.zeros((n,n))  # (i+1,j-1)
    dist2_2 = np.zeros((n,n))  # (i-1,j+1)
    dist2_3 = np.zeros((n,n))  # (i+1,j+1)

    # fill distance-0 neighbors
    m = mask1 & mask2
    dist0[m] = gY[m] + gX[m]

    # fill distance-1 neighbors
    m = mask1[1:, :] & mask2[:-1, :]
    dist1_0[1:, :][m] = 1 + gY[1:, :][m] + gX[:-1, :][m]

    m = mask1[:, 1:] & mask2[:, :-1]
    dist1_1[:, 1:][m] = 1 + gY[:, 1:][m] + gX[:, :-1][m]

    m = mask1[:-1, :] & mask2[1:, :]
    dist1_2[:-1, :][m] = 1 + gY[:-1, :][m] + gX[1:, :][m]

    m = mask1[:, :-1] & mask2[:, 1:]
    dist1_3[:, :-1][m] = 1 + gY[:, :-1][m] + gX[:, 1:][m]

    # fill distance-2 diagonals
    sqrt2 = np.sqrt(2)
    m = mask1[1:, 1:] & mask2[:-1, :-1]
    dist2_0[1:, 1:][m] = sqrt2 + gY[1:, 1:][m] + gX[:-1, :-1][m]

    m = mask1[:-1, 1:] & mask2[1:, :-1]
    dist2_1[:-1, 1:][m] = sqrt2 + gY[:-1, 1:][m] + gX[1:, :-1][m]

    m = mask1[1:, :-1] & mask2[:-1, 1:]
    dist2_2[1:, :-1][m] = sqrt2 + gY[1:, :-1][m] + gX[:-1, 1:][m]

    m = mask1[:-1, :-1] & mask2[1:, 1:]
    dist2_3[:-1, :-1][m] = sqrt2 + gY[:-1, :-1][m] + gX[1:, 1:][m]

    return TriDiagonal2D(
        dist0=dist0,
        dist1_0=dist1_0,
        dist1_1=dist1_1,
        dist1_2=dist1_2,
        dist1_3=dist1_3,
        dist2_0=dist2_0,
        dist2_1=dist2_1,
        dist2_2=dist2_2,
        dist2_3=dist2_3
    )


'''
Function to find the search direction
Parameters:
  pi: transportation plan
  grad: gradient of UOT
  M: upper bound for generalized simplex
  eps: tolerance
'''
def direction_p2(pi, grad, M, eps = 0.001):
  # Frank-Wolfe direction
  min_val = grad.min()
  if min_val < -eps:
    i_FW = grad.argmin()
  else:
    i_FW = (-1, -1, -1, -1)

  # Away Frank-Wolfe direction
  if not pi.has_positive():
    return i_FW, (-1, -1, -1, -1)

  max_val, i_AFW = grad.masked_argmax(pi, eps = 0.001)
  if (max_val <= eps):
    if (pi.sum() < M):
      return i_FW, (-1, -1, -1, -1)
    else:
      print("M: ", M, ", pi.sum(): ", pi.sum(), ". Increase M!")

  return i_FW, i_AFW


'''
Parameters:
  xk: transportation plan
  grad: gradient of UOT
  dir: indices of the search direction
  M: upper bound for generalized simplex
'''
def gap_calc_p2(xk, grad, dir, M):
  i_FW, _ = dir
  inner = grad.dot(xk)

  if i_FW[0] != -1:
    gap = - M * grad.get(*i_FW) + inner
  else:
    gap = inner

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
  if x1FW == -1:
    return (cost(x1AFW, x2AFW, y1AFW, y2AFW) + y_marg[y1AFW, y2AFW] +
            x_marg[x1AFW, x2AFW] - 2) / (1/mu[x1AFW, x2AFW] + 1/nu[y1AFW, y2AFW])
  elif x1AFW == -1:
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
def armijo_p2(x_marg, y_marg, grad, mu, nu, v, p, theta = 1, beta = 0.4, gamma = 0.5):
  # get the indices of the selected FW and AFW vertices
  FW, AFW = v

  if FW[0] != -1:
    inner = grad.get(*FW)
    if AFW[0] != -1:
      inner -= grad.get(*AFW)
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
      inner = -grad.get(*AFW)
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
def step_calc_p2(x_marg, y_marg, grad, mu, nu, v, p, step = "optimal", theta = 1, beta = 0.4, gamma = 0.5):
  if step == "armijo":
    return armijo_p2(x_marg, y_marg, grad, mu, nu, v, p, theta = theta, beta = beta, gamma = gamma)
  elif step == "optimal":
    return min(opt_step(x_marg, y_marg, mu, nu, v), theta)
  else:
    raise ValueError("Stepsize not recognized! Use 'armijo' or 'optimal'.")



"""
Update the gradient stored in a Tridiagonal2D object using only the non-zero entries.
Parameters:
  x_marg, y_marg  : X and Y marginals of the transportation plan
  grad            : gradient of UOT
  mask1, mask2    : masks for the zero coordinates
  v               : ((x1FW,x2FW,y1FW,y2FW), (x1AFW,x2AFW,y1AFW,y2AFW))
  """
def grad_update_p2(x_marg, y_marg, grad, mask1, mask2, v):
    n = grad.n
    (x1FW, x2FW, y1FW, y2FW), (x1AFW, x2AFW, y1AFW, y2AFW) = v

    # Helper: update 3x3 neighborhood around x (i0,j0)
    def update_x(i0, j0):
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                k = i0 + di
                l = j0 + dj
                if (0 <= k < n) and (0 <= l < n) and mask2[k,l]:
                    val = cost(i0, j0, k, l) + dUp_dx(x_marg[i0,j0], 2) + dUp_dx(y_marg[k,l], 2)
                    grad.set(i0, j0, k, l, val)

    # Helper: update 3x3 neighborhood around y (k0,l0)
    def update_y(k0, l0):
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                i = k0 + di
                j = l0 + dj
                if (0 <= i < n) and (0 <= j < n) and mask1[i,j]:
                    val = cost(i, j, k0, l0) + dUp_dx(x_marg[i,j], 2) + dUp_dx(y_marg[k0,l0], 2)
                    grad.set(i, j, k0, l0, val)

    # Update FW direction
    if x1FW != -1:
        update_x(x1FW, x2FW)
        update_y(y1FW, y2FW)

    # Update AFW direction
    if x1AFW != -1:
        update_x(x1AFW, x2AFW)
        update_y(y1AFW, y2AFW)

    return grad


'''
Pairwise Frank-Wolfe (only for p = 2)
Parameters:
  mu, nu: measures
  M: upper bound for generalized simplex
  step: stepsize calculation method
  max_iter: max iterations
  delta, eps: tolerance
'''
def PW_FW_dim2_p2(mu, nu, M, step = "optimal",
                  max_iter = 100, delta = 0.01, eps = 0.001):
  n = np.shape(mu)[0]
  # transportation plan, marginals and gradient initialization
  xk, x_marg, y_marg, mask1, mask2 = x_init_p2(mu, nu)
  grad_xk = grad_init_p2(x_marg, y_marg, mask1, mask2, n)

  for k in range(max_iter):

    # search direction vertices
    vk = direction_p2(xk, grad_xk, M, eps)

    # gap calculation
    gap = gap_calc_p2(xk, grad_xk, vk, M)

    if (gap <= delta) or (vk == ((-1,-1,-1,-1),(-1,-1,-1,-1))): 
      print("Converged after: ", k, " iterations ")
      return (xk, grad_xk, x_marg, y_marg)

    # coordinates + rows and columns update
    (x1FW, x2FW, y1FW, y2FW), (x1AFW, x2AFW, y1AFW, y2AFW) = vk
    if x1AFW != -1:

      gammak = step_calc_p2(x_marg, y_marg, grad_xk, mu, nu, vk, p = 2, step = step, theta = xk.get(x1AFW, x2AFW, y1AFW, y2AFW))

      xk.update(x1AFW, x2AFW, y1AFW, y2AFW, -gammak)
      x_marg[x1AFW, x2AFW] -= gammak / mu[x1AFW, x2AFW]
      y_marg[y1AFW, y2AFW] -= gammak / nu[y1AFW, y2AFW]
      if x1FW != -1:

        xk.update(x1FW, x2FW, y1FW, y2FW, gammak)
        x_marg[x1FW, x2FW] += gammak / mu[x1FW, x2FW]
        y_marg[y1FW, y2FW] += gammak / nu[y1FW, y2FW]
    else:
      # stepsize
      gammak = step_calc_p2(x_marg, y_marg, grad_xk, mu, nu, vk,
                         p = 2, step = step, theta = M - xk.sum() + xk.get(x1FW, x2FW, y1FW, y2FW))

      xk.update(x1FW, x2FW, y1FW, y2FW, gammak)
      x_marg[x1FW, x2FW] += gammak / mu[x1FW, x2FW]
      y_marg[y1FW, y2FW] += gammak / nu[y1FW, y2FW]

    # gradient update
    grad_xk = grad_update_p2(x_marg, y_marg, grad_xk, mask1, mask2, vk)

  print("Converged after: ", max_iter, " iterations ")
  return (xk, grad_xk, x_marg, y_marg)