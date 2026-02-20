import numpy as np
from FW_2dim_trunc import cost_matrix_trunc_dim2

# Set precision to 3 decimal places
np.set_printoptions(precision=3, suppress=True)

np.random.seed(0)
# Number of support point
n = 3  
R = 4
# Define two positive and discrete measures
mu = np.random.randint(0, 11, size = (n,n))
nu = np.random.randint(1, 101, size = (n,n))

print("cost matrix: ", cost_matrix_trunc_dim2(R))
