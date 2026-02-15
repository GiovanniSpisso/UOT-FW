import numpy as np

class TriDiagonal:
    def __init__(self, lower, main, upper):
        """
        lower[i] = A[i+1, i]
        main[i]   = A[i, i]
        upper[i] = A[i, i+1]
        """
        self.n = len(main)

        self.lower = lower   # length n-1
        self.main = main     # length n
        self.upper = upper   # length n-1

    # -----------------------------------------------------
    # Return A[i,j] (O(1))
    # -----------------------------------------------------
    def get(self, i, j):
        d = j - i
        if d == 0:
            return self.main[i]
        elif d == 1:
            return self.upper[i]
        elif d == -1:
            return self.lower[j]
        else:
            return 0.0

    # -----------------------------------------------------
    # Set A[i,j] = new_val in O(1)
    # -----------------------------------------------------
    def set(self, i, j, new_val):
        d = j - i

        # set the correct diagonal
        if d == 0:
            self.main[i] = new_val
        elif d == 1:
            self.upper[i] = new_val
        elif d == -1:
            self.lower[j] = new_val
        else:
            raise ValueError("Cannot set outside the 3-diagonal band.")

    # -----------------------------------------------------
    # Update A[i,j] += delta in O(1)
    # -----------------------------------------------------
    def update(self, i, j, delta):
        d = j - i

        if delta == 0:
            return

        # update the correct diagonal
        if d == 0:
            self.main[i] += delta
        elif d == 1:
            self.upper[i] += delta
        elif d == -1:
            self.lower[j] += delta
        else:
            raise ValueError("Cannot update outside the 3-diagonal band.")

    # -----------------------------------------------------
    # Calculate the sum of entries of A in O(n)
    # -----------------------------------------------------
    def sum(self):
      return self.main.sum() + self.upper.sum() + self.lower.sum()

    # -----------------------------------------------------
    # Return True if any stored entry is > 0 (off-band entries are zero).
    # -----------------------------------------------------
    def has_positive(self):
        if np.any(self.main > 0):
            return True
        if np.any(self.upper > 0):
            return True
        if np.any(self.lower > 0):
            return True
        return False

    # -----------------------------------------------------
    # Minimum over the whole n x n matrix (treat off-band entries as 0). Complexity O(n).
    # -----------------------------------------------------
    def min(self):
      return min(self.main.min(), self.upper.min(), self.lower.min(), 0)

    # -----------------------------------------------------
    # Return (i,j) of the minimum element in row-major order without building dense array. Complexity O(n).
    # -----------------------------------------------------
    def argmin(self):
        n = self.n
        # Find minimal value among the stored diagonals and its position
        min_val = np.inf
        min_pos = (-1, -1)

        # check in row-major order but only positions in the band (we must respect flattening order)
        for i in range(n):
            # check j = 0..n-1 but only band entries:
            # j = i-1
            if i >= 1:
                val = self.lower[i-1]
                if val < min_val:
                    min_val = val
                    min_pos = (i, i-1)
            # j = i
            val = self.main[i]
            if val < min_val:
                min_val = val
                min_pos = (i, i)
            # j = i+1
            if i + 1 < n:
                val = self.upper[i]
                if val < min_val:
                    min_val = val
                    min_pos = (i, i+1)

        # Note: min_val is min ONLY among band entries.
        return min_pos

    # -----------------------------------------------------
    # Given `mask` (another Tridiagonal object), return (max_val, (i,j)) among positions
    # where mask.get(i,j) is True. If no positions active, returns (-np.inf, (-1,-1)). Complexity O(n).
    # -----------------------------------------------------
    def masked_argmax(self, mask):
        n = self.n
        best_val = -np.inf
        best_pos = (-1, -1)

        # iterate band entries only
        for i in range(n):
            # (i,i) main
            if mask.get(i, i) > 0:
                v = self.main[i]
                if v > best_val:
                    best_val = v
                    best_pos = (i, i)
            # (i,i+1) upper
            if (i + 1 < n) and (mask.get(i, i+1) > 0):
                v = self.upper[i]
                if v > best_val:
                    best_val = v
                    best_pos = (i, i+1)
            # (i, i-1) lower (we can handle as (i, i-1) via get)
            if i - 1 >= 0 and mask.get(i, i-1):
                v = self.lower[i-1]
                if v > best_val:
                    best_val = v
                    best_pos = (i, i-1)

        return best_val, best_pos

    # -----------------------------------------------------
    # Compute sum_{i,j} self[i,j] * other[i,j], exploiting tridiagonal structure.  O(n).
    # -----------------------------------------------------
    def dot(self, other):
      return (
          np.dot(self.main,  other.main) +
          np.dot(self.upper, other.upper) +
          np.dot(self.lower, other.lower))

    # -----------------------------------------------------
    # Round the entries of A
    # -----------------------------------------------------
    def round(self, decimals=0):
      return TriDiagonal(
          np.round(self.lower, decimals),
          np.round(self.main, decimals),
          np.round(self.upper, decimals)
          )

    # -----------------------------------------------------
    # Convert to full dense matrix (O(n²))
    # -----------------------------------------------------
    def to_dense(self):
        A = np.zeros((self.n, self.n))
        for i in range(self.n):
            A[i, i] = self.main[i]
            if i + 1 < self.n:
                A[i, i+1] = self.upper[i]
                A[i+1, i] = self.lower[i]
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

"""
Compute the total UOT cost (p=2):
Parameters:
  pi: transportation plan
  x_marg, y_marg: X and Y marginals of the transportation plan
  c: cost function
  mu, nu: measures
"""
def cost_p2(pi, x_marg, y_marg, c, mu, nu):
  C1 = np.multiply(c, pi).sum()
  cost_row, cost_col = 0, 0

  cost_row = np.sum(mu * Up(x_marg, 2))
  cost_col = np.sum(nu * Up(y_marg, 2))

  C2 = cost_row + cost_col
  return C1 + C2


'''
Initial transportation plan + marginals (only for p = 2)
Parameters:
  mu, nu: measures
'''
def x_init_p2(mu, nu):
  den = mu + nu
  n = len(den)
  x0 = np.zeros(n, dtype=float)
  x_marg = np.zeros(n, dtype=float)
  y_marg = np.zeros(n, dtype=float)

  mask1 = (mu != 0)
  mask2 = (nu != 0)
  mask = mask1 & mask2

  x_marg[mask1] = 2 * nu[mask1] / den[mask1]
  y_marg[mask2] = 2 * mu[mask2] / den[mask2]
  x0[mask] = 2 * mu[mask] * nu[mask] / den[mask]

  x = TriDiagonal(np.zeros(n-1), x0, np.zeros(n-1))

  return x, x_marg, y_marg, mask1, mask2

'''
Function to define the gradient of UOT in O(n)
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  mask1, mask2: masks for the gradient
  c: cost function
  n: dimension
'''
def UOT_grad_p2(x_marg, y_marg, mask1, mask2, c, n):
  g_row = np.zeros(n)
  g_col = np.zeros(n)

  g_row[mask1] = dUp_dx(x_marg[mask1], 2)
  g_col[mask2] = dUp_dx(y_marg[mask2], 2)

  main = np.zeros(n)
  upper = np.zeros(n - 1)
  lower = np.zeros(n - 1)

  m = mask1 & mask2
  mask_upper = mask1[:-1] & mask2[1:]
  mask_lower = mask1[1:] & mask2[:-1]

  main[m] = g_row[m] + g_col[m]
  upper[mask_upper] = 1 + g_row[:-1][mask_upper] + g_col[1:][mask_upper]
  lower[mask_lower] = 1 + g_row[1:][mask_lower] + g_col[:-1][mask_lower]

  grad = TriDiagonal(lower, main, upper)

  return grad


'''
Function to find the search direction
Parameters:
  pi: transportation plan
  grad: gradient of UOT
  M: upper bound for generalized simplex
  eps: tolerance
'''
def direction_class(pi, grad, M, eps = 0.001):
    # -------- Frank-Wolfe direction --------
    min_val = grad.min()
    if min_val < - eps:
        FW_i, FW_j = grad.argmin()
    else:
        FW_i, FW_j = -1, -1

    # -------- Away direction / active set check --------
    # Active mask: we treat pi as tridiagonal, so mask True only on stored band entries > 0.
    if not pi.has_positive():
        return (FW_i, FW_j, -1, -1)

    # masked_grad: find maximum gradient among positions where pi > 0
    max_val, (AFW_i, AFW_j) = grad.masked_argmax(pi)

    # Decision logic same as before:
    # if max_val <= eps and total mass < M, return no away atom (classical FW)
    if (max_val <= eps):
      if (pi.sum() < M):
        return (FW_i, FW_j, -1, -1)
      else:
        print("M: ", M, ", pi.sum(): ", pi.sum(), ". Increase M!")

    return (FW_i, FW_j, AFW_i, AFW_j)


'''
Parameters:
  xk: transportation plan
  grad: gradient of UOT
  dir: indices of the search direction
  M: upper bound for generalized simplex
'''
def gap_calc_class(xk, grad, dir, M):
    # Frank-Wolfe point contributes only if it exists
    FW_i, FW_j, _, _ = dir
    inner = grad.dot(xk)
    if FW_i != -1:
        gap = - M * grad.get(FW_i, FW_j) + inner
    else:
        gap = inner

    return gap


'''
Optimal stepsize (p = 2)
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  c: cost function
  mu, nu: measures
  v: search direction
'''
def opt_step(x_marg, y_marg, c, mu, nu, v):
  xFW, yFW, xAFW, yAFW = v
  if xFW == -1:
    return (c[xAFW, yAFW] + y_marg[yAFW] + x_marg[xAFW] - 2) / (1/mu[xAFW] + 1/nu[yAFW])
  elif xAFW == -1:
    return (2 - c[xFW, yFW] - y_marg[yFW] - x_marg[xFW]) / (1/mu[xFW] + 1/nu[yFW])
  elif xFW == xAFW:
    return (c[xFW, yAFW] - c[xFW, yFW] + y_marg[yAFW] - y_marg[yFW]) / (1/nu[yAFW] + 1/nu[yFW])
  elif yFW == yAFW:
    return (c[xAFW, yFW] - c[xFW, yFW] + x_marg[xAFW] - x_marg[xFW]) / (1/mu[xAFW] + 1/mu[xFW])
  else:
    return (c[xAFW, yAFW] - c[xFW, yFW] + y_marg[yAFW] - y_marg[yFW] + x_marg[xAFW]
            - x_marg[xFW]) / (1/mu[xAFW] + 1/mu[xFW] + 1/nu[yAFW] + 1/nu[yFW])

'''
Armijo stepsize
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  grad: gradient of UOT
  mu, nu: measures
  v: search direction
  c: cost function
  p: main parameter that defines the p-entropy
  theta, beta, gamma: parameters for the Armijo stepsize
'''
def armijo_class(x_marg, y_marg, grad, mu, nu, v, c, p = 2, theta = 1, beta = 0.4, gamma = 0.5):
  # get the indices of the selected FW and AFW vertices
  FW_i, FW_j, AFW_i, AFW_j = v

  if FW_i != -1:
    inner = grad.get(FW_i, FW_j)
    if AFW_i != -1:
      inner -= grad.get(AFW_i, AFW_j)
      diff = (theta * (c[FW_i, FW_j] - c[AFW_i, AFW_j]) + (Up(x_marg[FW_i] + theta/mu[FW_i], p) - Up(x_marg[FW_i], p))*mu[FW_i] +
              (Up(y_marg[FW_j] + theta/nu[FW_j], p) - Up(y_marg[FW_j], p))*nu[FW_j] + (Up(x_marg[AFW_i] - theta/mu[AFW_i], p) - Up(x_marg[AFW_i], p))*mu[AFW_i]
              + (Up(y_marg[AFW_j] - theta/nu[AFW_j], p) - Up(y_marg[AFW_j], p))*nu[AFW_j])
      while diff > beta*theta*inner:
        theta = gamma * theta
        diff = (theta * (c[FW_i, FW_j] - c[AFW_i, AFW_j]) + (Up(x_marg[FW_i] + theta/mu[FW_i], p) - Up(x_marg[FW_i], p))*mu[FW_i] +
                (Up(y_marg[FW_j] + theta/nu[FW_j], p) - Up(y_marg[FW_j], p))*nu[FW_j] + (Up(x_marg[AFW_i] - theta/mu[AFW_i], p) - Up(x_marg[AFW_i], p))*mu[AFW_i]
                + (Up(y_marg[AFW_j] - theta/nu[AFW_j], p) - Up(y_marg[AFW_j], p))*nu[AFW_j])
    else:
      diff = (theta*c[FW_i, FW_j] + (Up(x_marg[FW_i] + theta/mu[FW_i], p) - Up(x_marg[FW_i], p))*mu[FW_i] +
              (Up(y_marg[FW_j] + theta/nu[FW_j], p) - Up(y_marg[FW_j], p))*nu[FW_j])
      while diff > beta*theta*inner:
        theta = gamma * theta
        diff = (theta*c[FW_i, FW_j] + (Up(x_marg[FW_i] + theta/mu[FW_i], p) - Up(x_marg[FW_i], p))*mu[FW_i] +
                (Up(y_marg[FW_j] + theta/nu[FW_j], p) - Up(y_marg[FW_j], p))*nu[FW_j])

  elif AFW_i != -1:
      inner = -grad.get(AFW_i, AFW_j)
      diff = (- theta * c[AFW_i, AFW_j] + (Up(x_marg[AFW_i] - theta/mu[AFW_i], p) - Up(x_marg[AFW_i], p))*mu[AFW_i]
              + (Up(y_marg[AFW_j] - theta/nu[AFW_j], p) - Up(y_marg[AFW_j], p))*nu[AFW_j])
      while diff > beta*theta*inner:
        theta = gamma * theta
        diff = (- theta * c[AFW_i, AFW_j] + (Up(x_marg[AFW_i] - theta/mu[AFW_i], p) - Up(x_marg[AFW_i], p))*mu[AFW_i]
                + (Up(y_marg[AFW_j] - theta/nu[AFW_j], p) - Up(y_marg[AFW_j], p))*nu[AFW_j])

  return theta


'''
Stepsize calculation
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  grad: gradient of UOT
  mu, nu: measures
  v: indices of the selected FW and AFW vertices
  c: cost matrix
  p: main parameter
  theta, beta, gamma: Armijo parameters
'''
def step_calc_class(x_marg, y_marg, grad, mu, nu, v, c, p, step = "optimal", theta = 1, beta = 0.4, gamma = 0.5):
  if step == "armijo":
    return armijo_class(x_marg, y_marg, grad, mu, nu, v, c, p, theta = theta, beta = beta, gamma = gamma)
  elif step == "optimal":
    return min(opt_step(x_marg, y_marg, c, mu, nu, v), theta)
  else:
    raise ValueError("Stepsize not recognized! Use 'armijo' or 'optimal'.")


'''
Function to update the gradient of UOT
Parameters:
  x_marg, y_marg  : X and Y marginals of the transportation plan
  grad            : gradient of UOT
  mask1, mask2    : masks for the gradient
  c               : cost function
  v               : descent direction
'''
def UOT_grad_update_p2(x_marg, y_marg, grad, mask1, mask2, c, v):
    n = grad.n
    FW_i, FW_j, AFW_i, AFW_j = v

    def update_row(i):
      # (i, i) main diagonal
      if mask2[i]:
        grad.set(i, i, c[i, i] + dUp_dx(x_marg[i], 2) + dUp_dx(y_marg[i], 2))

      # (i, i+1) upper
      if (i + 1 < n) and mask2[i+1]:
        grad.set(i, i+1, c[i, i+1] + dUp_dx(x_marg[i], 2) + dUp_dx(y_marg[i+1], 2))

      # (i, i-1) lower
      if (i - 1 >= 0) and mask2[i-1]:
        grad.set(i, i-1, c[i, i-1] + dUp_dx(x_marg[i], 2) + dUp_dx(y_marg[i-1], 2))


    def update_col(j):
      # (j, j) main
      if mask1[j]:
         grad.set(j, j, c[j, j] + dUp_dx(x_marg[j], 2) + dUp_dx(y_marg[j], 2))

      # (j-1, j) lower
      if (j - 1 >= 0) and mask1[j-1]:
         grad.set(j-1, j, c[j-1, j] + dUp_dx(x_marg[j-1], 2) + dUp_dx(y_marg[j], 2))

      # (j+1, j) upper
      if (j + 1 < n) and mask1[j+1]:
         grad.set(j+1, j, c[j+1, j] + dUp_dx(x_marg[j+1], 2) + dUp_dx(y_marg[j], 2))

    # Update FW row/column if needed
    if FW_i != -1:
        update_row(FW_i)
        update_col(FW_j)

    # Update AWF row/column if needed
    if AFW_i != -1:
        update_row(AFW_i)
        update_col(AFW_j)

    return grad


'''
Pairwise Frank-Wolfe (p = 2)
Parameters:
  mu, nu: measures
  c: cost function
  M: upper bound for generalized simplex
  max_iter: max iterations
  delta, eps: tolerance
'''

def PW_FW_dim1_p2(mu, nu, c, M,
                  max_iter = 100, delta = 0.01, eps = 0.001):
  n = np.shape(mu)[0]

  # transportation plan, marginals and gradient initialization
  xk, x_marg, y_marg, mask1, mask2 = x_init_p2(mu, nu)
  grad_xk = UOT_grad_p2(x_marg, y_marg, mask1, mask2, c, n)

  for k in range(max_iter):
    # search direction
    vk = direction_class(xk, grad_xk, M, eps)

    # gap calculation
    gap = gap_calc_class(xk, grad_xk, vk, M)

    if (gap <= delta) or (vk == (-1,-1,-1,-1)):
      print("Converged after: ", k, " iterations ")
      return xk, grad_xk, x_marg, y_marg

    # coordinates + rows and columns update
    xFW, yFW, xAFW, yAFW = vk

    if xAFW != -1:
      gamma0 = xk.get(xAFW, yAFW) - 1e-10
      gammak = min(opt_step(x_marg, y_marg, c, mu, nu, vk), gamma0)
      xk.update(xAFW, yAFW, -gammak)
      x_marg[xAFW] -= gammak / mu[xAFW]
      y_marg[yAFW] -= gammak / nu[yAFW]
      if xFW != -1:
        xk.update(xFW, yFW, gammak)
        x_marg[xFW] += gammak / mu[xFW]
        y_marg[yFW] += gammak / nu[yFW]
    else:
      gamma0 = M - np.sum(x_marg * mu) + xk.get(xFW, yFW)
      gammak = min(opt_step(x_marg, y_marg, c, mu, nu, vk), gamma0)
      xk.update(xFW, yFW, gammak)
      x_marg[xFW] += gammak / mu[xFW]
      y_marg[yFW] += gammak / nu[yFW]

    # gradient update
    grad_xk = UOT_grad_update_p2(x_marg, y_marg, grad_xk, mask1, mask2, c, vk)

  print("Converged after: ", max_iter, " iterations ")
  return xk, grad_xk, x_marg, y_marg