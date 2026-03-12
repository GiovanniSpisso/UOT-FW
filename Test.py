import numpy as np
from itertools import combinations

terms = [1, 2, 3]

print(sum(np.prod(combo) for combo in combinations(terms, 0)))