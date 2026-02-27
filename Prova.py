import numpy as np
import numpy.ma as ma
import FW_1dim_p2 as fw_p2

# Parameters
n = 5  # problem size
p = 2  # entropy parameter (p=2)

# Tolerance parameters
delta = 0.001
eps = 0.001
max_iter = 10

# Generate test data
np.random.seed(0)
mu = np.random.randint(0, 5, size=n).astype(float)
nu = np.random.randint(0, 5, size=n).astype(float)

mu = np.ma.masked_equal(mu, 0)
nu = np.ma.masked_equal(nu, 0)

# Set M (upper bound for generalized simplex)
M = n * (np.sum(mu) + np.sum(nu))

print(f"Problem setup:")
print(f"  n = {n}")
print(f"  p = {p}")
print(f"  M (upper bound) = {M:.2f}")
print(f"  max_iter = {max_iter}")
print(f"  delta (convergence tol) = {delta}")
print(f"  eps (direction tol) = {eps}")
print(f"  mu = {mu}")
print(f"  nu = {nu}")
print()

# Initialize
xk, x_marg, y_marg, mask_3n = fw_p2.x_init_p2(mu, nu, n)
print("xk: ", xk)
print("x_marg: ", x_marg)
print("y_marg: ", y_marg)
mat = fw_p2.vec_to_mat_p2(xk, n)
print("xk as matrix:\n", mat)

grad_xk = fw_p2.grad_p2(x_marg, y_marg, n, mask_3n)
print("Gradient at xk: ", grad_xk)
mat = fw_p2.vec_to_mat_p2(grad_xk, n)
print("xk as matrix:\n", mat)