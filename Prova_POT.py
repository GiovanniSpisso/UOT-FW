import numpy as np
import ot
import pandas as pd

#np.random.seed(0)
#n = 5
#eps = 0.001
#
#a = np.random.randint(1, 1000, size=n)
#b = np.random.randint(1, 1000, size=n)
## Correct - 2D column vector
#X = np.arange(n, dtype=np.float64).reshape(-1, 1)   # shape (n, 1)
#Y = np.arange(n, dtype=np.float64).reshape(-1, 1)   # shape (n, 1)
#M = ot.dist(X, Y)  # now works, gives (n, m) cost matrix
#M /= M.max()
#reg_kl = 1
#
#print(np.round(ot.sinkhorn_unbalanced(a, b, M, eps, reg_kl), 9))

df1 = pd.read_csv("DOTmark_1.0/data512_1001.csv", header=None)
df2 = pd.read_csv("DOTmark_1.0/data512_1002.csv", header=None)

np.random.seed(0)
n = 512
eps = 0.001

#a = np.random.randint(1, 100, size=(n,n)).ravel()
#b = np.random.randint(1, 100, size=(n,n)).ravel()
a = df1.values.ravel()  # Flatten the 2D array to 1D
b = df2.values.ravel()  # Flatten the 2D array to 1D
# Correct - 2D column vector
X = np.array(np.meshgrid(np.arange(n), np.arange(n))).T.reshape(-1, 2)
Y = np.array(np.meshgrid(np.arange(n), np.arange(n))).T.reshape(-1, 2)
M = ot.dist(X, Y)  # now works, gives (n, m) cost matrix
M /= M.max()
reg_kl = 1

print(np.round(ot.sinkhorn_unbalanced(a, b, M, eps, reg_kl), 9))
