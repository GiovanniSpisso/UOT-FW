import numpy as np
from FW_2dim_p2 import to_dense_dim2_p2, x_init_dim2_p2, grad_dim2_p2, \
    LMO_dim2_p2, gap_calc_dim2_p2, cost_ind_dim2_p2, opt_step_dim2_p2

# Set precision to 3 decimal places
np.set_printoptions(precision=3, suppress=True)

np.random.seed(0)
# Number of support point
n = 2  
# Define two positive and discrete measures
#mu = np.random.randint(0, 11, size = (n,n))
#nu = np.random.randint(1, 101, size = (n,n))
mu = np.array([[0.5, 2], [0.5, 2]])
nu = np.array([[1, 0.1], [1, 0.1]])

x0, x_marg, y_marg, mask1, mask2 = x_init_dim2_p2(mu, nu, n)
grad = grad_dim2_p2(x_marg, y_marg, mask1, mask2, n)

print("x0: \n", x0)
#print("mask1: ", mask1)
#print("mask2: ", mask2)
print("grad: \n", grad)

print("Dense transportation plan: \n", to_dense_dim2_p2(x0, n))
print("x_marg: ", x_marg*mu)
print("y_marg: ", y_marg*nu)
print("Dense gradient: \n", to_dense_dim2_p2(grad, n))

dir = LMO_dim2_p2(x0, grad, M = 1000)

print("FW and AFW direction: ", dir)

mat_idx = (dir[0][0][0], dir[1][0][0])
full = (dir[0][1], dir[1][1])
gamma = opt_step_dim2_p2(x_marg, y_marg, mu, nu, mat_idx, full)

print("Optimal step size: ", gamma)

sum_term = np.sum(x0 * grad)
print("Sum term: ", sum_term)