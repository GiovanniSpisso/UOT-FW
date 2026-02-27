import sys
import numpy as np
import cProfile
import pstats

from FW_1dim import PW_FW_dim1, UOT_cost
from FW_1dim_p2 import PW_FW_dim1_p2
from FW_1dim_p1_5 import PW_FW_dim1_p1_5
from FW_1dim_trunc import PW_FW_dim1_trunc, truncated_cost
from FW_2dim import PW_FW_dim2
from FW_2dim_p2 import PW_FW_dim2_p2
from FW_2dim_trunc import PW_FW_dim2_trunc, truncated_cost_dim2, cost_matrix_trunc_dim2


def make_data(n):
      # 1D measures: size n
      mu1 = np.random.randint(0, 100, size=n)
      nu1 = np.random.randint(0, 100, size=n)
      c1 = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))

      # 2D measures: size n x n
      mu2 = np.random.randint(1, 100, size=(n, n))
      nu2 = np.random.randint(1, 100, size=(n, n))

      return mu1, nu1, c1, mu2, nu2


def profile_call(name, func, args_pos=(), args_kw=None, stats_n=5):
      print('\n' + '='*60)
      print('Profiling:', name)
      print('='*60)
      profiler = cProfile.Profile()
      profiler.enable()
      try:
            if args_kw is None:
                  res = func(*args_pos)
            else:
                  res = func(*args_pos, **args_kw)
      except Exception as e:
            profiler.disable()
            print(f'Error while running {name}:', e)
            return None
      profiler.disable()

      stats = pstats.Stats(profiler)
      stats.sort_stats('tottime')
      stats.print_stats(stats_n)
      return res


if __name__ == '__main__':
      # Allow separate n values for 1D and 2D solvers. Defaults: n_1d=10, n_2d=10
      n_1d = int(sys.argv[1]) if len(sys.argv) > 1 else 10
      n_2d = int(sys.argv[2]) if len(sys.argv) > 2 else 10
      max_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
      p_generic = float(sys.argv[4]) if len(sys.argv) > 4 else 2  # p for FW_1dim and FW_2dim
      R = int(sys.argv[5]) if len(sys.argv) > 5 else 3  # truncation radius for FW_truncated
      delta = 0.01
      eps = 0.001
      np.random.seed(0)

      # Create 1D data
      mu1, nu1, c, _, _ = make_data(n_1d)
      M1 = n_1d * (np.sum(mu1) + np.sum(nu1))
      
      # Create 2D data
      _, _, _, mu2, nu2 = make_data(n_2d)
      M2 = n_2d * (np.sum(mu2) + np.sum(nu2))
      
      # Create truncated cost vector for FW_truncated
      c_trunc = np.concatenate([
            np.full(n_1d - abs(k), abs(k))
            for k in range(-R + 1, R)
      ])

      # Build 2D cost tensor: c[i,j,k,l] = sqrt((i-j)^2 + (k-l)^2)
      idx = np.arange(n_2d)
      di = (idx[:, None] - idx[None, :]) ** 2
      dj = (idx[:, None] - idx[None, :]) ** 2
      c2d = np.sqrt(di[:, None, :, None] + dj[None, :, None, :])

      # List of solvers to profile: (display_name, callable, pos_args_tuple, kw_args_dict)
      solvers = [
            ('FW_2dim_p2', PW_FW_dim2_p2, (mu2, nu2, M2), {'max_iter': max_iter, 'delta': delta, 'eps': eps}),
            ('FW_2dim', PW_FW_dim2, (mu2, nu2, M2, p_generic, c2d), {'max_iter': max_iter, 'delta': delta, 'eps': eps}),
            ('FW_2dim_trunc', PW_FW_dim2_trunc, (mu2, nu2, M2, p_generic, R), {'max_iter': max_iter, 'delta': delta, 'eps': eps}),
            ('FW_1dim', PW_FW_dim1, (mu1, nu1, M1, p_generic, c), {'max_iter': max_iter, 'delta': delta, 'eps': eps}),
            ('FW_1dim_p2', PW_FW_dim1_p2, (mu1, nu1, M1), {'max_iter': max_iter, 'delta': delta, 'eps': eps}),
            ('FW_1dim_p1_5', PW_FW_dim1_p1_5, (mu1, nu1, M1), {'max_iter': max_iter, 'delta': delta, 'eps': eps}),
            ('FW_truncated', PW_FW_dim1_trunc, (mu1, nu1, M1, p_generic, c_trunc, R), {'max_iter': max_iter, 'delta': delta, 'eps': eps}),
      ]

      results = {}
      for name, func, pos, kw in solvers:
            # Run each solver and print top-5 slowest functions
            result = profile_call(name, func, args_pos=pos, args_kw=kw, stats_n=5)
            results[name] = result

      print('\n' + '='*60)
      print('DIMENSION 1: COST COMPARISON')
      print('='*60)
      
      if results.get('FW_1dim') and results.get('FW_1dim_p2') and results.get('FW_1dim_p1_5'):
            xk_general, grad_general, x_marg_general, y_marg_general = results['FW_1dim']
            xk_p2, grad_p2, x_marg_p2, y_marg_p2 = results['FW_1dim_p2']
            xk_p1_5, grad_p1_5, x_marg_p1_5, y_marg_p1_5 = results['FW_1dim_p1_5']
            
            cost_general = UOT_cost(xk_general, x_marg_general, y_marg_general, c, mu1, nu1, p_generic)
            
            from FW_1dim_p2 import cost_p2
            from FW_1dim_p1_5 import cost_p1_5
            cost_p2_val = cost_p2(xk_p2, x_marg_p2, y_marg_p2, mu1, nu1)
            cost_p1_5_val = cost_p1_5(xk_p1_5, x_marg_p1_5, y_marg_p1_5, mu1, nu1)
            
            print(f"\nFinal cost (FW_1dim):    {cost_general:.10f}")
            print(f"Final cost (FW_1dim_p2): {cost_p2_val:.10f}")
            print(f"Final cost (FW_1dim_p1_5): {cost_p1_5_val:.10f}")
      
      if results.get('FW_truncated'):
            xk_trunc, grad_trunc, x_marg_trunc, y_marg_trunc, s_i, s_j = results['FW_truncated']
            cost_trunc = truncated_cost(xk_trunc, x_marg_trunc, y_marg_trunc, c_trunc, mu1, nu1, p_generic, s_i, s_j, R)
            print(f"Final cost (FW_1dim_trunc): {cost_trunc:.10f}")
      
      print('\n' + '='*60)
      print('DIMENSION 2: COST COMPARISON')
      print('='*60)
      
      if results.get('FW_2dim') and results.get('FW_2dim_p2'):
            from FW_2dim import UOT_cost
            from FW_2dim_p2 import cost_dim2_p2
            
            xk_2d, grad_2d, x_marg_2d, y_marg_2d = results['FW_2dim']
            xk_2d_p2, grad_2d_p2, x_marg_2d_p2, y_marg_2d_p2 = results['FW_2dim_p2']
            
            cost_2d_general = UOT_cost(xk_2d, x_marg_2d, y_marg_2d, c2d, mu2, nu2, p_generic)
            cost_2d_p2_val = cost_dim2_p2(xk_2d_p2, x_marg_2d_p2, y_marg_2d_p2, mu2, nu2)
            
            print(f"\nFinal cost (FW_2dim):    {cost_2d_general:.10f}")
            print(f"Final cost (FW_2dim_p2): {cost_2d_p2_val:.10f}")

      if results.get('FW_2dim_trunc'):
            xk_2d_trunc, grad_2d_trunc, x_marg_2d_trunc, y_marg_2d_trunc, s_i_2d_trunc, s_j_2d_trunc = results['FW_2dim_trunc']
            c2d_trunc, _ = cost_matrix_trunc_dim2(R)
            cost_2d_trunc_val = truncated_cost_dim2(xk_2d_trunc, x_marg_2d_trunc, y_marg_2d_trunc, c2d_trunc,
                                                    mu2, nu2, p_generic, s_i_2d_trunc, s_j_2d_trunc, R)
            print(f"Final cost (FW_2dim_trunc): {cost_2d_trunc_val:.10f}")
            
      print('\n' + '='*60)
      print('All profiling runs completed.\n')