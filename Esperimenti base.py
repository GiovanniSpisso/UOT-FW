import numpy as np
from FW_2dim_p2 import x_init_dim2_p2, grad_dim2_p2

np.random.seed(0)
# Number of support point
n = 2  
# Define two positive and discrete measures
mu = np.random.randint(0, 11, size = (n,n))
nu = np.random.randint(0, 11, size = (n,n))

x0, x_marg, y_marg, mask1, mask2 = x_init_dim2_p2(mu, nu, n)
grad = grad_dim2_p2(x_marg, y_marg, mask1, mask2, n)


print("x0: ", x0)
print("x_marg: ", x_marg)
print("y_marg: ", y_marg)
print("mask1: ", mask1)
print("mask2: ", mask2)
print("grad: ", grad)
