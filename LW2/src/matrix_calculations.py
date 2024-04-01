import numpy as np
from numbers import Number
from typing import Iterable


# /\ = min(x, y)
# \/ = x * y

class MatrixGenerators:

    @classmethod
    def _matrixA(cls, p: int, m: int):
        return np.matrix(
            (1 + 1) * np.random.rand(m, p) - 1
        ).tolist()

    @classmethod
    def _matrixB(cls, m: int, q: int):
        return np.matrix(
            (1 + 1) * np.random.rand(q, m) - 1
        ).tolist()

    @classmethod
    def _matrixE(cls, m: int):
        return np.matrix(
            (1 + 1) * np.random.rand(m) - 1
        ).tolist()

    @classmethod
    def _matrixG(cls, p: int, q: int):
        return np.matrix(
            (1 + 1) * np.random.rand(q, p) - 1
        ).tolist()
    
    @classmethod
    def generate_matrices(cls, p: int, m: int, q: int):
        return {
            'A': cls._matrixA(p, m),
            'B': cls._matrixB(m, q),
            'E': cls._matrixE(m),
            'G': cls._matrixG(p, q)
        }


class ComputationUnit:

    def __init__(self, config: dict) -> None:
        self._tacts = 0
        # Specifying vector size                
        self._MAX_VEC_SIZE = config.get('MAX_VEC_SIZE', 4)

        # Specifying operation computing time
        self._ADD_TIME = config.get('ADD_TIME', 1)
        self._SUB_TIME = config.get('SUB_TIME', 1)
        self._MUL_TIME = config.get('MUL_TIME', 1)
        self._DIV_TIME = config.get('DIV_TIME', 1)
        self._CPR_TIME = config.get('CPR_TIME', 1)
    
        # Specifying of used operations during use of PU
        self._used_add = 0
        self._used_sub = 0
        self._used_mul = 0
        self._used_div = 0
        self._used_cpr = 0
    
    # Simple operations
    def add(self, x: Number, y: Number) -> Number:
        self._used_add += 1
        return x + y
    
    def sub(self, x: Number, y: Number) -> Number:
        self._used_sub += 1
        return x - y
    
    def mul(self, x: Number, y: Number) -> Number:
        self._used_mul += 1
        return x * y
    
    def div(self, x: Number, y: Number) -> Number:
        self._used_div += 1
        return x / y

    def cpr(self, x: Number, y: Number) -> Number:
        # 1 - X greater that Y
        # 0 - X equals to Y
        # -1 - X less that Y
        self._used_cpr += 1
        if x < y:
            return -1
        elif x == y:
            return 0
        else:
            return 1


class ComplexCU:
    '''Specified computating unit containing
    complex operations (conjunction, disjunction, implicance)'''

    def __init__(self, comp_unit: ComputationUnit) -> None:
        self._cu = comp_unit
    
    def min(self, *args) -> Number | None:
        if len(args) == 0:
            return None
        min = args[0]
        for num in args[1:]:
            if self._cu.cpr(num, min) == -1:
                min = num
        return min
    
    def max(self, *args) -> Number | None:
        if len(args) == 0:
            return None
        max = args[0]
        for num in args[1:]:
            if self._cu.cpr(num, max) == 1:
                max = num
        return max
    
    def conj(self, x: Number, y: Number) -> Number:
        return self.min(x, y)

    def disj(self, x: Number, y: Number) -> Number:
        return self._cu.mul(x, y)

    def arr_conj(self, arr: Iterable[Number]) -> Number:
        val = 1
        for num in arr:
            val = self._cu.mul(val, num)
        return val

    def arr_disj(self, arr: Iterable[Number]) -> Number:
        val = 1
        for num in arr:
            val = self._cu.mul(val, self._cu.sub(1, num))
        val = self._cu.sub(1, val)
        return val

    def aconj(self, arr: Iterable[Number]) -> Number:
        return self.arr_conj(arr)

    def adisj(self, arr: Iterable[Number]) -> Number:
        return self.arr_disj(arr)
