import numpy as np
from numbers import Number
from typing import Iterable


# /\ = min(x, y)
# \/ = x * y
# x ~> y = max(1 - x, y)

class MatrixGenerators:

    @classmethod
    def _matrixA(cls, p: int, m: int):
        return ((1 + 1) * np.random.rand(p, m)
                - 1).tolist()

    @classmethod
    def _matrixB(cls, m: int, q: int):
        return ((1 + 1) * np.random.rand(m, q)
                - 1).tolist()

    @classmethod
    def _matrixE(cls, m: int):
        return ((1 + 1) * np.random.rand(m)
                - 1).tolist()

    @classmethod
    def _matrixG(cls, p: int, q: int):
        return ((1 + 1) * np.random.rand(p, q)
                - 1).tolist()
    
    @classmethod
    def generate_matrices(cls, p: int, m: int, q: int):
        return {
            'A': cls._matrixA(p, m),
            'B': cls._matrixB(m, q),
            'E': cls._matrixE(m),
            'G': cls._matrixG(p, q)
        }


class ComputationUnit:
    '''Simple computating operations: addition,
    subtraction, multiplying, dividing, comparing'''

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
    
    def impl(self, x: Number, y: Number) -> Number:
        return self.max(self._cu.sub(1, x), y)


class MatrixPU:

    def __init__(self, config: dict) -> None:
        # Generating matrices
        self._p = config.get('p', 2)
        self._m = config.get('m', 2)
        self._q = config.get('q', 2)
        matrices = MatrixGenerators.generate_matrices(
            self._p, self._m, self._q            
        )
        self._matrix_a = matrices['A']
        self._matrix_b = matrices['B']
        self._matric_e = matrices['E']
        self._matrix_g = matrices['G']

        print(matrices)        

        # Assigning computation units
        self._pu = ComputationUnit(config)
        self._complex_pu = ComplexCU(self._pu)

        self._matrix_c = None
        self._matrix_f = None
        self._matrix_d = None

    def compute_d(self):
        self._matrix_d = [
            [self._compute_d_elem(i, j)
             for j in range(self._q)]
             for i in range(self._p)
        ]
    
    def _compute_d_elem(self, i: int, j: int) -> Number:
        arr: list[Number] = [
            self._complex_pu.conj(self._matrix_a[i][k],
                                  self._matrix_b[k][j])
            for k in range(self._m)
        ]
        return self._complex_pu.arr_disj(arr) 
    
    def compute_f(self):
        self._matrix_f = [
            [
                [
                    self._compute_f_elem(i, j, k)
                    for k in range(self._m)
                ]
                for j in range(self._q)
            ]
            for i in range(self._p)
        ]

    def _compute_f_elem(self, i: int, j: int, k: int) -> Number:
        # Get elements for slightly better performance
        aik: Number = self._matrix_a[i][k]
        bkj: Number = self._matrix_b[k][j]
        ek: Number = self._matric_e[k]

        # Computing terms because expression is big
        # Left part
        t1: Number = self._complex_pu.impl(aik, bkj)
        t2: Number = self._pu.sub(self._pu.mul(2, ek), 1)
        t3: Number = self._pu.mul(t1, t2)
        t_left = self._pu.mul(t3, ek)

        # Right part
        t1: Number = self._complex_pu.impl(bkj, aik)
        t2: Number = self._complex_pu.impl(aik, bkj)
        t3: Number = self._pu.sub(self._pu.mul(4, t2), 2)
        t4: Number = self._pu.mul(t3, ek)
        t5: Number = self._pu.add(1, t4)
        t6: Number = self._pu.sub(1, ek)
        t7: Number = self._pu.mul(t1, t5)
        t_right: Number = self._pu.mul(t7, t6)

        return self._pu.add(
            t_left, t_right
        )
    
    def compute_c(self):
        if self._matrix_f is None:
            self.compute_f()
        if self._matrix_d is None:
            self.compute_d()
        self._matrix_c = [
            [self._compute_c_elem(i, j) for j in range(self._q)]
            for i in range(self._p)
        ]
    
    def _compute_c_elem(self, i: int, j: int):
        # Get elements for slightly better performance
        fij: list[Number] = self._matrix_f[i][j]
        gij: Number = self._matrix_g[i][j]
        dij: Number = self._matrix_d[i][j]

        # Computing terms because expression is big
        # Left part
        t1: Number = self._complex_pu.arr_conj(fij)
        t2: Number = self._pu.sub(self._pu.mul(3, gij), 2)
        t3: Number = self._pu.mul(t1, t2)
        t_left: Number = self._pu.sub(t3, gij)

        # Right part
        t1: Number = self._pu.sub(1, gij)
        t2: Number = self._complex_pu.arr_conj(fij)
        t3: Number = self._complex_pu.disj(t2, dij)
        t4: Number = self._pu.mul(4, t3)
        t5: Number = self._pu.mul(3, dij)
        t6: Number = self._pu.sub(t4, t5)
        t7: Number = self._pu.mul(t6, gij)
        t8: Number = self._pu.add(dij, t7)
        t_right: Number = self._pu.mul(t1, t8)

        return self._pu.add(t_left, t_right)

    @property
    def matrix_c(self):
        if self._matrix_c is None:
            self.compute_c()
        return self._matrix_c    
