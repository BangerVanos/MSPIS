from typing import Iterable
from numbers import Number
from itertools import chain
from src.x_matrix_calculations import MatrixPU


def reduction_add(vec: Iterable[Number]) -> Number:
        neutral = 0
        new_vec = vec
        while len(new_vec := [new_vec[i] + (new_vec[i + 1] if i + 1 < len(new_vec) else neutral) for i in range(0, len(new_vec), 2)]) > 1:            
            print(new_vec)
        print(new_vec)            
        return new_vec[0]

print(reduction_add([1, 2, 3, 4, 5, 8]))
print(list(chain.from_iterable([[1, 2], [[1, 2, 3]]])))

pu = MatrixPU({'p': 2, 'm': 3, 'q': 4})
pu.compute_c()
print(pu.matrices)
    