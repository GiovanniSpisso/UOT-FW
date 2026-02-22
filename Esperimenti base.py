import numpy as np
from FW_2dim_trunc import cost_matrix_trunc_dim2, x_init_trunc_dim2, grad_trunc_dim2
from FW_2dim_p2 import x_init_dim2_p2, grad_dim2_p2, LMO_dim2_p2, gap_calc_dim2_p2, \
    opt_step_dim2_p2, grad_update_dim2_p2, apply_step_dim2_p2, update_sum_term_dim2_p2, cost_dim2_p2

# Set precision to 3 decimal places
np.set_printoptions(precision=3, suppress=True)

np.random.seed(0)
# Number of support point
n = 2  
R = 2
p = 2
# Define two positive and discrete measures
mu = np.random.randint(0, 11, size = (n,n))
nu = np.random.randint(1, 101, size = (n,n))
print("mu = ", mu)
print("nu = ", nu)

c_trunc = cost_matrix_trunc_dim2(R)
print("cost matrix trunc: ", c_trunc)

x_trunc, x_marg_trunc, y_marg_trunc, mask1_trunc, mask2_trunc = x_init_trunc_dim2(mu, nu, n, R, p)
x, x_marg, y_marg, mask1, mask2 = x_init_dim2_p2(mu, nu, n)
print("x_trunc = \n", x_trunc)
print("x = \n", x)
print("x_marg_trunc = \n", x_marg_trunc)
print("x_marg = \n", x_marg)
print("y_marg_trunc = \n", y_marg_trunc)
print("y_marg = \n", y_marg)

grad_trunc = grad_trunc_dim2(x_marg_trunc, y_marg_trunc, mask1_trunc, mask2_trunc, c_trunc, p, n, R)
grad = grad_dim2_p2(x_marg, y_marg, mask1, mask2, n)
print("grad_trunc = \n", grad_trunc)
print("grad = \n", grad)