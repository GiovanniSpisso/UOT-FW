import numpy as np

class SevenDiagonal:
    """
    7-diagonal matrix:
        lower3[i] = A[i+3, i]
        lower2[i] = A[i+2, i]
        lower1[i] = A[i+1, i]
        main[i]   = A[i, i]
        upper1[i] = A[i, i+1]
        upper2[i] = A[i, i+2]
        upper3[i] = A[i, i+3]
    """

    def __init__(self, lower3, lower2, lower1, main, upper1, upper2, upper3):
        self.n = len(main)

        self.lower3 = lower3   # len n-3
        self.lower2 = lower2   # len n-2
        self.lower1 = lower1   # len n-1
        self.main   = main     # len n
        self.upper1 = upper1   # len n-1
        self.upper2 = upper2   # len n-2
        self.upper3 = upper3   # len n-3

    # =====================================================
    # GET ENTRY (i,j) in O(1)
    # =====================================================
    def get(self, i, j):
        d = j - i
        if d == 0:
            return self.main[i]
        elif d == 1:
            return self.upper1[i]
        elif d == 2:
            return self.upper2[i]
        elif d == 3:
            return self.upper3[i]
        elif d == -1:
            return self.lower1[j]
        elif d == -2:
            return self.lower2[j]
        elif d == -3:
            return self.lower3[j]
        else:
            return 0.0

    # =====================================================
    # SET ENTRY (i,j) = new_val
    # =====================================================
    def set(self, i, j, new_val):
        d = j - i
        if d == 0:
            self.main[i] = new_val
        elif d == 1:
            self.upper1[i] = new_val
        elif d == 2:
            self.upper2[i] = new_val
        elif d == 3:
            self.upper3[i] = new_val
        elif d == -1:
            self.lower1[j] = new_val
        elif d == -2:
            self.lower2[j] = new_val
        elif d == -3:
            self.lower3[j] = new_val
        else:
            raise ValueError("Cannot set outside the 7-diagonal band.")

    # =====================================================
    # UPDATE ENTRY += delta
    # =====================================================
    def update(self, i, j, delta):
        if delta == 0:
            return
        d = j - i
        if d == 0:
            self.main[i] += delta
        elif d == 1:
            self.upper1[i] += delta
        elif d == 2:
            self.upper2[i] += delta
        elif d == 3:
            self.upper3[i] += delta
        elif d == -1:
            self.lower1[j] += delta
        elif d == -2:
            self.lower2[j] += delta
        elif d == -3:
            self.lower3[j] += delta
        else:
            raise ValueError("Cannot update outside the 7-diagonal band.")

    # =====================================================
    # SUM ALL STORED ENTRIES
    # =====================================================
    def sum(self):
        return (self.main.sum() + self.upper1.sum() + self.upper2.sum() + self.upper3.sum() +
                self.lower1.sum() + self.lower2.sum() + self.lower3.sum())

    # -----------------------------------------------------
    # TRUE IF ANY ENTRY IS > 0
    # -----------------------------------------------------
    def has_positive(self):
        return (
            np.any(self.main  > 0) or
            np.any(self.upper1 > 0) or
            np.any(self.upper2 > 0) or
            np.any(self.upper3 > 0) or
            np.any(self.lower1 > 0) or
            np.any(self.lower2 > 0) or
            np.any(self.lower3 > 0)
        )

    # =====================================================
    # MINIMUM VALUE (considering off-band entries = 0)
    # =====================================================
    def min(self):
        return min(
            0,
            self.main.min(), self.upper1.min(), self.upper2.min(), self.upper3.min(),
            self.lower1.min(), self.lower2.min(), self.lower3.min()
        )

    # =====================================================
    # ARGMIN IN ROW-MAJOR ORDER, scanning only band entries
    # =====================================================
    def argmin(self):
        n = self.n
        min_val = float("inf")
        min_pos = (-1, -1)

        for i in range(n):
            # (i,j) where j = i-3..i+3
            for d in (-3, -2, -1, 0, 1, 2, 3):
                j = i + d
                if 0 <= j < n:
                    v = self.get(i, j)
                    if v < min_val:
                        min_val = v
                        min_pos = (i, j)

        return min_pos

    # =====================================================
    # masked_argmax(mask):
    #   Return (max_val, (i,j)) over positions where
    #   mask.get(i,j) > 0, scanning only the 7 diagonals.
    #   If nothing active: returns (-inf, (-1, -1)).
    # =====================================================
    def masked_argmax(self, mask):
        n = self.n
        best_val = -np.inf
        best_pos = (-1, -1)

        for i in range(n):
            # d = j - i ∈ {-3,-2,-1,0,1,2,3}
            for d in (-3, -2, -1, 0, 1, 2, 3):
                j = i + d
                if 0 <= j < n and mask.get(i, j) > 0:
                    v = self.get(i, j)
                    if v > best_val:
                        best_val = v
                        best_pos = (i, j)

        return best_val, best_pos

    # =====================================================
    # DOT PRODUCT WITH ANOTHER SevenDiagonal
    # =====================================================
    def dot(self, other):
        return (
            np.dot(self.main,   other.main) +
            np.dot(self.upper1, other.upper1) +
            np.dot(self.upper2, other.upper2) +
            np.dot(self.upper3, other.upper3) +
            np.dot(self.lower1, other.lower1) +
            np.dot(self.lower2, other.lower2) +
            np.dot(self.lower3, other.lower3)
        )

    # =====================================================
    # ROUND ALL DIAGONALS
    # =====================================================
    def round(self, decimals=0):
        return SevenDiagonal(
            np.round(self.lower3, decimals),
            np.round(self.lower2, decimals),
            np.round(self.lower1, decimals),
            np.round(self.main,   decimals),
            np.round(self.upper1, decimals),
            np.round(self.upper2, decimals),
            np.round(self.upper3, decimals)
        )

    # =====================================================
    # CONVERT TO FULL MATRIX (O(n²))
    # =====================================================
    def to_dense(self):
        A = np.zeros((self.n, self.n))
        for i in range(self.n):
            A[i, i] = self.main[i]

            if i + 1 < self.n:
                A[i, i+1] = self.upper1[i]
                A[i+1, i] = self.lower1[i]

            if i + 2 < self.n:
                A[i, i+2] = self.upper2[i]
                A[i+2, i] = self.lower2[i]

            if i + 3 < self.n:
                A[i, i+3] = self.upper3[i]
                A[i+3, i] = self.lower3[i]

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
Compute the total UOT cost:
Parameters:
  pi: transportation plan
  x_marg, y_marg: X and Y marginals of the transportation plan
  c: cost function
  mu, nu: measures
  p: main parameter that defines the p-entropy
"""
def UOT_cost(pi, x_marg, y_marg, c, mu, nu, p):
  C1 = np.multiply(c, pi).sum()
  cost_row, cost_col = 0, 0

  cost_row = np.sum(mu * Up(x_marg, p))
  cost_col = np.sum(nu * Up(y_marg, p))

  C2 = cost_row + cost_col
  return C1 + C2


'''
Initial transportation plan + marginals (only for p = 1.5)
Parameters:
  mu, nu: measures
'''
def x_init_p1_5(mu, nu):
  n = len(mu)
  x_marg = np.zeros(n)
  y_marg = np.zeros(n)

  mask1 = (mu != 0)
  mask2 = (nu != 0)
  mask = mask1 & mask2

  x0 = ((mu[mask] * nu[mask]) / (mu[mask]**0.5 + nu[mask]**0.5)**2) * 4
  x_marg[mask1] = x0 / mu[mask1]
  y_marg[mask2] = x0 / nu[mask2]

  x = SevenDiagonal(
        np.zeros(n-3),
        np.zeros(n-2),
        np.zeros(n-1),
        x0,
        np.zeros(n-1),
        np.zeros(n-2),
        np.zeros(n-3)
    )

  return x, x_marg, y_marg, mask1, mask2


'''
Function to define the gradient of UOT in O(n)
Parameters:
  x_marg, y_marg: X and Y marginals of the transportation plan
  mask1, mask2: masks for the gradient
  c: cost function
  n: dimension
'''
def UOT_grad_p1_5(x_marg, y_marg, mask1, mask2, c, n):
  g_row = np.zeros(n)
  g_col = np.zeros(n)

  g_row[mask1] = dUp_dx(x_marg[mask1], 1.5)
  g_col[mask2] = dUp_dx(y_marg[mask2], 1.5)

  # Allocate 7 diagonals
  main  = np.zeros(n)
  u1 = np.zeros(n-1)
  u2 = np.zeros(n-2)
  u3 = np.zeros(n-3)
  l1 = np.zeros(n-1)
  l2 = np.zeros(n-2)
  l3 = np.zeros(n-3)

  m = mask1 & mask2
  mask_u1 = mask1[:-1] & mask2[1:]
  mask_l1 = mask1[1:] & mask2[:-1]
  mask_u2 = mask1[:-2] & mask2[2:]
  mask_l2 = mask1[2:] & mask2[:-2]
  mask_u3 = mask1[:-3] & mask2[3:]
  mask_l3 = mask1[3:] & mask2[:-3]

  main[m] = g_row[m] + g_col[m]
  u1[mask_u1] = 1 + g_row[:-1][mask_u1] + g_col[1:][mask_u1]
  l1[mask_l1] = 1 + g_row[1:][mask_l1] + g_col[:-1][mask_l1]
  u2[mask_u2] = 2 + g_row[:-2][mask_u2] + g_col[2:][mask_u2]
  l2[mask_l2] = 2 + g_row[2:][mask_l2] + g_col[:-2][mask_l2]
  u3[mask_u3] = 3 + g_row[:-3][mask_u3] + g_col[3:][mask_u3]
  l3[mask_l3] = 3 + g_row[3:][mask_l3] + g_col[:-3][mask_l3]

  grad = SevenDiagonal(l3, l2, l1, main, u1, u2, u3)

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
def armijo_class(x_marg, y_marg, grad, mu, nu, v, c, p = 1.5, theta = 1, beta = 0.4, gamma = 0.5):
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
Function to update the gradient of UOT
Parameters:
  x_marg, y_marg  : X and Y marginals of the transportation plan
  grad            : gradient of UOT
  mask1, mask2    : masks for the gradient
  c               : cost function
  v               : descent direction
'''
def UOT_grad_update_p1_5(x_marg, y_marg, grad, mask1, mask2, c, v):
    n = grad.n
    FW_i, FW_j, AFW_i, AFW_j = v

    def update_row(i):
      g_x = dUp_dx(x_marg[i], 1.5)
      # (i, i) main diagonal
      if mask2[i]:
        grad.set(i, i, c[i, i] + g_x + dUp_dx(y_marg[i], 1.5))

      # upper bands: (i, i+1), (i, i+2), (i, i+3)
      for d in (1, 2, 3):
        j = i + d
        if (j < n) and mask2[j]:
            grad.set(i, j, c[i, j] + g_x + dUp_dx(y_marg[j], 1.5))

      # lower bands: (i, i-1), (i, i-2), (i, i-3)
      for d in (1, 2, 3):
        j = i - d
        if (j >= 0) and mask2[j]:
            grad.set(i, j, c[i, j] + g_x + dUp_dx(y_marg[j], 1.5))

    def update_col(j):
      g_y = dUp_dx(y_marg[j], 1.5)
      # (j, j) main
      grad.set(j, j, c[j, j] + dUp_dx(x_marg[j], 1.5) + g_y)

      # upper bands in column: (j-1, j), (j-2, j), (j-3, j)
      for d in (1, 2, 3):
        i = j - d
        if (i >= 0) and mask1[i]:
            grad.set(i, j, c[i, j] + dUp_dx(x_marg[i], 1.5) + g_y)

      # lower bands in column: (j+1, j), (j+2, j), (j+3, j)
      for d in (1, 2, 3):
        i = j + d
        if (i < n) and mask1[i]:
            grad.set(i, j, c[i, j] + dUp_dx(x_marg[i], 1.5) + g_y)

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
Pairwise Frank-Wolfe (p = 1.5)
Parameters:
  mu, nu: measures
  c: cost function
  M: upper bound for generalized simplex
  max_iter: max iterations
  step: stepsize ["Optimal", "Armijo"]
  delta, eps: tolerance
'''

def PW_FW_dim1_p1_5(mu, nu, c, M,
                    max_iter = 100, delta = 0.01, eps = 0.001):
  n = np.shape(mu)[0]
  # transportation plan, marginals and gradient initialization
  xk, x_marg, y_marg, mask1, mask2 = x_init_p1_5(mu, nu)
  grad_xk = UOT_grad_p1_5(x_marg, y_marg, mask1, mask2, c, n)

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
      gammak = armijo_class(x_marg, y_marg, grad_xk, mu, nu, vk, c, p = 1.5, theta = gamma0)
      xk.update(xAFW, yAFW, -gammak)
      x_marg[xAFW] -= gammak / mu[xAFW]
      y_marg[yAFW] -= gammak / nu[yAFW]
      if xFW != -1:
        xk.update(xFW, yFW, gammak)
        x_marg[xFW] += gammak / mu[xFW]
        y_marg[yFW] += gammak / nu[yFW]
    else:
      gamma0 = M - np.sum(x_marg * mu) + xk.get(xFW, yFW)
      gammak = armijo_class(x_marg, y_marg, grad_xk, mu, nu, vk, c, p = 1.5, theta = gamma0)
      xk.update(xFW, yFW, gammak)
      x_marg[xFW] += gammak / mu[xFW]
      y_marg[yFW] += gammak / nu[yFW]

    grad_xk = UOT_grad_update_p1_5(x_marg, y_marg, grad_xk, mask1, mask2, c, vk)

  print("Converged after: ", max_iter, " iterations ")
  return xk, grad_xk, x_marg, y_marg