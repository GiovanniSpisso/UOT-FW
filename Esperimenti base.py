import numpy as np
from FW_2dim_trunc import cost_matrix_trunc_dim2, x_init_trunc_dim2, grad_trunc_dim2, \
    truncated_cost_dim2, LMO_trunc_dim2_x, LMO_trunc_dim2_s, gap_calc_trunc_dim2
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
mu = np.array([[0, 1], [0.2, 0.8]])
nu = np.array([[1, 0.5], [2, 0.2]])
print("mu = ", mu)
print("nu = ", nu)

M = 2 * (np.sum(mu) + np.sum(nu))
eps = 0.001

c_trunc, displacement_map = cost_matrix_trunc_dim2(R)
print("cost matrix trunc: ", c_trunc)
print("displacement map: ", displacement_map)

x_trunc, x_marg_trunc, y_marg_trunc, mask1_trunc, mask2_trunc = x_init_trunc_dim2(mu, nu, n, R, p)
x, x_marg, y_marg, mask1, mask2 = x_init_dim2_p2(mu, nu, n)
print("x_trunc = \n", x_trunc)
print("x = \n", x)
print("x_marg_trunc = \n", x_marg_trunc)
print("x_marg = \n", x_marg)
print("y_marg_trunc = \n", y_marg_trunc)
print("y_marg = \n", y_marg)

grad_trunc, (grad_si, grad_sj) = grad_trunc_dim2(x_marg_trunc, y_marg_trunc, mask1_trunc, mask2_trunc, c_trunc, displacement_map, p, n, R)
grad = grad_dim2_p2(x_marg, y_marg, mask1, mask2, n)
print("grad_trunc = \n", grad_trunc)
print("grad_si = \n", grad_si)
print("grad_sj = \n", grad_sj)
print("grad = \n", grad)

s_i, s_j = np.zeros((n,n)), np.zeros((n,n))

truncated_cost = truncated_cost_dim2(x_trunc, x_marg_trunc, y_marg_trunc, c_trunc, 
                                     mu, nu, p, s_i, s_j, R)
cost = cost_dim2_p2(x, x_marg, y_marg, mu, nu)
print("truncated cost = ", truncated_cost)
print("cost = ", cost)

i_FW_trunc, i_AFW_trunc = LMO_trunc_dim2_x(x_trunc, grad_trunc, displacement_map, M)
print("i_FW_trunc = ", i_FW_trunc)
print("i_AFW_trunc = ", i_AFW_trunc)
i_FW, i_AFW = LMO_dim2_p2(x, grad, M)
print("i_FW = ", i_FW)
print("i_AFW = ", i_AFW)
FW_si, FW_sj, AFW_si, AFW_sj = LMO_trunc_dim2_s(s_i, s_j, (grad_si, grad_sj), M, eps, mask1_trunc, mask2_trunc)
print("si_FW = ", FW_si)
print("si_AFW = ", AFW_si)
print("sj_FW = ", FW_sj)
print("sj_AFW = ", AFW_sj)