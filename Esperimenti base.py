import numpy as np

x = np.array([0,1,2,3,4,5,6,7,8])
v = 5

print(x[v])
print(x[(v+3)%9])
print(x[(v+6)%9]) 