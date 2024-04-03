# Методы решения задач в интеллектуальных системах
# Лабораторная работа №2 Вариант 7
# Авторы: Заломов Р.А., Готин И.А.
# Дата: 03.04.24
# Данный файл содержит классы, отвечающие за
# генерирование матриц, обсчёт матриц, просчёт параметров
# параллельной и последовательной архитектур


import numpy as np
from numbers import Number
from typing import Iterable
from math import ceil, prod
import numpy as np


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


class MatrixUtility:

    @classmethod
    def shape(cls, matrix: Iterable | Number) -> tuple[int, ...]:
        if isinstance (matrix, Iterable):
            outermost_size = len(matrix)
            row_shape = cls.shape(matrix[0])
            return (outermost_size, *row_shape)
        else:
            return ()


class ComputationUnit:
    '''Simple computating operations: addition,
    subtraction, multiplying, dividing, comparing'''

    def __init__(self, config: dict = {}) -> None:       
    
        # Specifying of used operations during use of PU
        self._used_add = 0
        self._used_sub = 0
        self._used_mul = 0
        self._used_div = 0
        self._used_cpr = 0

        # Specifying operation computing time
        self._ADD_TIME = config.get('ADD_TIME', 1)
        self._SUB_TIME = config.get('SUB_TIME', 1)
        self._MUL_TIME = config.get('MUL_TIME', 1)
        self._DIV_TIME = config.get('DIV_TIME', 1)
        self._CPR_TIME = config.get('CPR_TIME', 1)

        # Specifying vector size                
        self._PROCS_ELEMS = config.get('PROCS_ELEMS', 4)

        # Tacts done (only sequential architecture)        
        self._tacts = 0

        # Simulation of parallel tacts
        self._par_tacts = 0

        # Max Vec Size (in elems)
        self._MAX_VEC_SIZE = config.get('MAX_VEC_SIZE', 4)

        # Average Program Length Numerator
        self._lavg_numerator = 0 
    
    # Simple operations
    def add(self, x: Number, y: Number) -> Number:
        self._used_add += 1
        self._lavg_numerator += 2 * self._ADD_TIME
        self._tacts += self._ADD_TIME
        return x + y
    
    def sub(self, x: Number, y: Number) -> Number:
        self._used_sub += 1
        self._lavg_numerator += 2 * self._SUB_TIME
        self._tacts += self._SUB_TIME
        return x - y
    
    def mul(self, x: Number, y: Number) -> Number:
        self._used_mul += 1
        self._lavg_numerator += 2 * self._MUL_TIME
        self._tacts += self._MUL_TIME
        return x * y
    
    def div(self, x: Number, y: Number) -> Number:
        self._used_div += 1
        self._lavg_numerator += 2 * self._DIV_TIME
        self._tacts += self._DIV_TIME
        return x / y

    def cpr(self, x: Number, y: Number) -> Number:
        # 1 - X greater that Y
        # 0 - X equals to Y
        # -1 - X less that Y
        self._used_cpr += 1
        self._lavg_numerator += 2 * self._CPR_TIME
        self._tacts += self._CPR_TIME
        if x < y:
            return -1
        elif x == y:
            return 0
        else:
            return 1
    
    def add_par_tacts(self, par_tacts: Number) -> None:
        self._par_tacts += par_tacts  
    
    @property
    def add_time(self) -> Number:
        return self._ADD_TIME
    
    @property
    def sub_time(self) -> Number:
        return self._SUB_TIME
    
    @property
    def mul_time(self) -> Number:
        return self._ADD_TIME
    
    @property
    def div_time(self) -> Number:
        return self._DIV_TIME
    
    @property
    def cpr_time(self) -> Number:
        return self._CPR_TIME
    
    @property
    def procs_elems(self) -> Number:
        return self._PROCS_ELEMS

    @property
    def vec_size(self) -> Number:
        return self._MAX_VEC_SIZE
    
    @property
    def tacts(self) -> Number:
        return self._tacts
    
    @property
    def par_tacts(self) -> Number:
        return self._par_tacts
    
    @property
    def report(self) -> dict:
        return {
            'used_add': self._used_add,
            'used_sub': self._used_sub,
            'used_mul': self._used_mul,
            'used_div': self._used_div,
            'used_cpr': self._used_cpr,
            'length_avg': self._lavg_numerator,
            # 'seq_tacts': self._tacts,
            # 'par_tacts': self._par_tacts            
        }


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


class ComputeReport:

    def __init__(self, pu: ComputationUnit) -> None:              

        # Need ComputationUnit object to make report
        self._pu = pu    
    
    def _count_seq_tacts(self) -> Number:
        return self._pu.tacts
    
    def _count_par_tacts(self) -> Number:
        return self._pu.par_tacts
    
    def _count_acceleration_coeff(self) -> Number:
        seq_tacts = self._pu.tacts
        par_tacts = self._pu.par_tacts
        return seq_tacts / par_tacts
    
    def _count_efficency(self) -> Number:
        acceleration_coef = self._count_acceleration_coeff()
        return acceleration_coef / self._pu.procs_elems

    
    @property
    def report(self) -> dict:
        cu_report = self._pu.report
        seq_tacts = self._count_seq_tacts()
        par_tacts = self._count_par_tacts()
        print(seq_tacts, par_tacts)
        acceleration_coeff = self._count_acceleration_coeff()
        efficency = self._count_efficency()        
        report = {
            'seq_tacts': seq_tacts,
            'par_tacts': par_tacts,
            'acceleration_coeff': acceleration_coeff,
            'efficency': efficency,
            'length': par_tacts                               
        }
        report.update(cu_report)
        return report


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

        # Assigning computation units
        self._pu = ComputationUnit(config)
        self._complex_pu = ComplexCU(self._pu)

        self._matrix_c = None
        self._matrix_f = None
        self._matrix_d = None

        # Report unit to provide users with reports
        # on processing unit params (does not include matrices)
        self._reporter: ComputeReport = ComputeReport(self._pu)

    def compute_d(self):                
        self._matrix_d = [
            [self._compute_d_elem(i, j)
             for j in range(self._q)]
             for i in range(self._p)
        ]
                         
    
    def _compute_d_elem(self, i: int, j: int) -> Number:
        old_tacts = self._pu.tacts
        arr: list[Number] = [
            self._complex_pu.conj(self._matrix_a[i][k],
                                  self._matrix_b[k][j])
            for k in range(self._m)
        ]
        
        arr_disj = self._complex_pu.arr_disj(arr)
        self._pu.add_par_tacts(ceil((self._pu.tacts - old_tacts) / min(self._pu.vec_size, self._pu.procs_elems)))
             
        return arr_disj
    
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

        old_tacts = self._pu.tacts

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

        # Final term
        t = self._pu.add(t_left, t_right)

        self._pu.add_par_tacts(ceil((self._pu.tacts - old_tacts) / min(self._pu.vec_size, self._pu.procs_elems)))

        return t
    
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

        old_tacts = self._pu.tacts 

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

        # Final term
        t = self._pu.add(t_left, t_right)

        self._pu.add_par_tacts(ceil((self._pu.tacts - old_tacts) / min(self._pu.vec_size, self._pu.procs_elems)))        

        return t

    @property
    def matrix_c(self):
        if self._matrix_c is None:
            self.compute_c()
        return self._matrix_c
    
    @property
    def data_matrices(self):
        return (
            self._matrix_a,
            self._matrix_b,
            self._matric_e,
            self._matrix_g,
            # self._matrix_d,
            # self._matrix_f
        )

    @property
    def matrices(self) -> dict:
        if self._matrix_c is None:
            self.compute_c()        
        return {
                'matrix_A': self._matrix_a,
                'matrix_B': self._matrix_b,
                'matrix_E': self._matric_e,
                'matrix_G': self._matrix_g,
                'matrix_D': self._matrix_d,
                'matrix_F': self._matrix_f,
                'matrix_C': self._matrix_c,                
            }
    
    @property
    def report(self) -> dict:
        matrices = self.matrices
        prog_rank = sum([prod(MatrixUtility.shape(matrix))
                         for matrix in self.data_matrices])
        report = self._reporter.report
        lavg = report['length_avg'] / prog_rank
        divergence_coef = report['length'] / lavg        
        report.update(
            {
                'length_avg': lavg,
                'divergence_coeff': divergence_coef,
                'rank': prog_rank 
            }
        )
        report.update(matrices)
        return report
