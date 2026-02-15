import sys
import numpy as np
import cProfile
import pstats

#from FW_2dim_p2 import PW_FW_dim2_p2
#from FW_2dim import PW_FW_dim2
from FW_1dim import PW_FW_dim1, UOT_cost
from FW_1dim_p2 import PW_FW_dim1_p2
from FW_1dim_p1_5 import PW_FW_dim1_p1_5


def make_data(n, seed=0):
      np.random.seed(seed)
      # 1D measures: size n
      mu1 = np.random.randint(0, 100, size=n)
      nu1 = np.random.randint(0, 100, size=n)
      c1 = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))

      # 2D measures: size n x n
      mu2 = np.random.randint(1, 1001, size=(n, n))
      nu2 = np.random.randint(1, 1001, size=(n, n))

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
      delta = 0.001
      eps = 0.001

      # Create 1D data
      mu1, nu1, c, _, _ = make_data(n_1d)
      M1 = n_1d * (np.sum(mu1) + np.sum(nu1))
      
      # Create 2D data
      _, _, _, mu2, nu2 = make_data(n_2d)
      M2 = n_2d * (np.sum(mu2) + np.sum(nu2))

      # List of solvers to profile: (display_name, callable, pos_args_tuple, kw_args_dict)
      solvers = [
            #('FW_2dim_p2', PW_FW_dim2_p2, (mu2, nu2, M2), {'step': 'optimal', 'max_iter': max_iter, 'delta': delta, 'eps': eps}),
            #('FW_2dim', PW_FW_dim2, (mu2, nu2, M2, p_generic), {'step': 'armijo', 'max_iter': max_iter, 'delta': delta, 'eps': eps}),
            ('FW_1dim', PW_FW_dim1, (mu1, nu1, M1, p_generic, c), {'max_iter': max_iter, 'delta': delta, 'eps': eps}),
            ('FW_1dim_p2', PW_FW_dim1_p2, (mu1, nu1, M1), {'max_iter': max_iter, 'delta': delta, 'eps': eps}),
            ('FW_1dim_p1_5', PW_FW_dim1_p1_5, (mu1, nu1, M1), {'max_iter': max_iter, 'delta': delta, 'eps': eps}),
      ]

      results = {}
      for name, func, pos, kw in solvers:
            # Run each solver and print top-5 slowest functions
            result = profile_call(name, func, args_pos=pos, args_kw=kw, stats_n=5)
            results[name] = result

      print('\n' + '='*60)
      print('COST COMPARISON: FW_1dim vs FW_1dim_p2')
      print('='*60)
      
      if results.get('FW_1dim') and results.get('FW_1dim_p2'):
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
            
      print('\n' + '='*60)
      print('All profiling runs completed.\n')