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
from itertools import chain


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
    subtraction, multiplying, dividing, max, min'''

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

    def max(self, x: Number, y: Number) -> Number:
        self._used_cpr += 1
        self._lavg_numerator += 2 * self._CPR_TIME
        self._tacts += self._CPR_TIME
        return x if x > y else y 

    def min(self, x: Number, y: Number) -> Number:
        self._used_cpr += 1
        self._lavg_numerator += 2 * self._CPR_TIME
        self._tacts += self._CPR_TIME
        return x if x < y else y      
    
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
    def tacts(self) -> Number:
        return self._tacts    
    
    @property
    def report(self) -> dict:
        return {
            'used_add': self._used_add,
            'used_sub': self._used_sub,
            'used_mul': self._used_mul,
            'used_div': self._used_div,
            'used_cpr': self._used_cpr,
            'length_avg': self._lavg_numerator,
            'tacts': self._tacts           
        }


class ComplexCU:
    '''Specified computating unit containing
    complex operations (conjunction, disjunction, implicance)'''

    def __init__(self, config: dict = {}) -> None:
        self._procs = [ComputationUnit(config) for
                       _ in range(config.get('PROCS_ELEMS', 4))]
        self._procs_amount = config.get('PROCS_ELEMS', 4)
        
    def add(self, vec_1: Iterable[Number],
            vec_2: Iterable[Number]) -> Iterable[Number]:
        if not len(vec_1) == len(vec_2):
            raise ValueError('Vec sizes must be equal in SIMD operations!')
        return [self._procs[i % self._procs_amount].add(vec_1[i], vec_2[i])
                for i in range(len(vec_1))]
    
    def sub(self, vec_1: Iterable[Number],
            vec_2: Iterable[Number]) -> Iterable[Number]:
        if not len(vec_1) == len(vec_2):
            raise ValueError('Vec sizes must be equal in SIMD operations!')
        return [self._procs[i % self._procs_amount].sub(vec_1[i], vec_2[i])
                for i in range(len(vec_1))]
    
    def mul(self, vec_1: Iterable[Number],
            vec_2: Iterable[Number]) -> Iterable[Number]:
        if not len(vec_1) == len(vec_2):
            raise ValueError('Vec sizes must be equal in SIMD operations!')
        return [self._procs[i % self._procs_amount].mul(vec_1[i], vec_2[i])
                for i in range(len(vec_1))]
    
    def div(self, vec_1: Iterable[Number],
            vec_2: Iterable[Number]) -> Iterable[Number]:
        if not len(vec_1) == len(vec_2):
            raise ValueError('Vec sizes must be equal in SIMD operations!')
        return [self._procs[i % self._procs_amount].div(vec_1[i], vec_2[i])
                for i in range(len(vec_1))]
    
    def max(self, vec_1: Iterable[Number],
            vec_2: Iterable[Number]) -> Iterable[Number]:
        if not len(vec_1) == len(vec_2):
            raise ValueError('Vec sizes must be equal in SIMD operations!')
        return [self._procs[i % self._procs_amount].max(vec_1[i], vec_2[i])
                for i in range(len(vec_1))]
    
    def min(self, vec_1: Iterable[Number],
            vec_2: Iterable[Number]) -> Iterable[Number]:
        if not len(vec_1) == len(vec_2):
            raise ValueError('Vec sizes must be equal in SIMD operations!')
        return [self._procs[i % self._procs_amount].min(vec_1[i], vec_2[i])
                for i in range(len(vec_1))]
    
    # def reduction_add(self, vec: Iterable[Number]) -> Number:
    #     neutral = 0
    #     new_vec = vec
    #     while len(new_vec := [self._procs[ind % self._procs_amount].add(
    #         new_vec[i], (new_vec[i + 1] if i + 1 < len(new_vec) else neutral)
    #     ) for ind, i in enumerate(range(0, len(new_vec), 2))]) > 1:
    #         continue
    #     return new_vec[0]    
    
    # def reduction_mul(self, vec: Iterable[Number]) -> Number:
    #     neutral = 1
    #     new_vec = vec
    #     while len(new_vec := [self._procs[ind % self._procs_amount].mul(
    #         new_vec[i], (new_vec[i + 1] if i + 1 < len(new_vec) else neutral)
    #     ) for ind, i in enumerate(range(0, len(new_vec), 2))]) > 1:
    #         continue
    #     return new_vec[0]
    
    def reduction_add(self, vec: Iterable[Number]) -> Number:
        neutral = 0
        splits = [(vec[i], vec[i + 1] if i + 1 < len(vec) else neutral)
                  for i in range(0, len(vec), 2)]
        new_vec = vec
        while len(new_vec) != 1:
            new_vec = [self._procs[ind % self._procs_amount].add(*split)
                       for ind, split in enumerate(splits)]
            splits = [(new_vec[i], new_vec[i + 1] if i + 1 < len(new_vec) else neutral)
                      for i in range(0, len(new_vec), 2)]
        return new_vec[0]
    
    def reduction_mul(self, vec: Iterable[Number]) -> Number:
        neutral = 1
        splits = [(vec[i], vec[i + 1] if i + 1 < len(vec) else neutral)
                  for i in range(0, len(vec), 2)]
        new_vec = vec
        while len(new_vec) != 1:
            new_vec = [self._procs[ind % self._procs_amount].mul(*split)
                       for ind, split in enumerate(splits)]
            splits = [(new_vec[i], new_vec[i + 1] if i + 1 < len(new_vec) else neutral)
                      for i in range(0, len(new_vec), 2)]            
        return new_vec[0]
    
    def conj(self, vec_1: Iterable[Number],
             vec_2: Iterable[Number]) -> Iterable[Number]:
        if not len(vec_1) == len(vec_2):
            raise ValueError('Vec sizes must be equal in SIMD operations!')
        return self.min(vec_1, vec_2)
    
    def disj(self, vec_1: Iterable[Number],
             vec_2: Iterable[Number]) -> Iterable[Number]:
        if not len(vec_1) == len(vec_2):
            raise ValueError('Vec sizes must be equal in SIMD operations!')
        return self.mul(vec_1, vec_2)
    
    def comlex_conj(self, vec: Iterable[Iterable[Number]]) -> Iterable[Number]:
        reductions = [self.reduction_mul(elem) for elem in vec]
        return reductions
    
    def complex_disj(self, vec: Iterable[Iterable[Number]]) -> Iterable[Number]:        
        reductions = [self.reduction_mul(self.sub([1] * len(elem), elem))
                      for elem in vec]
        return self.sub([1] * len(reductions), reductions)
    
    def cconj(self, vec: Iterable[Iterable[Number]]) -> Iterable[Number]:
        return self.comlex_conj(vec)
    
    def cdisj(self, vec: Iterable[Iterable[Number]]) -> Iterable[Number]:
        return self.complex_disj(vec)
    
    def impl(self, vec_1: Iterable[Number],
             vec_2: Iterable[Number]) -> Iterable[Number]:
        if not len(vec_1) == len(vec_2):
            raise ValueError('Vec sizes must be equal in SIMD operations!')
        return self.max(self.sub([1] * len(vec_1), vec_1), vec_2)
    
    @property
    def report(self):
        pu_reports = [pu.report for pu in self._procs]
        seq_tacts = sum([pu.tacts for pu in self._procs])
        par_tacts = max([pu.tacts for pu in self._procs])

        acceleration_coeff = seq_tacts / par_tacts
        efficency = acceleration_coeff / self._procs_amount

        return {
            'used_add': sum([rep['used_add'] for rep in pu_reports]),
            'used_sub': sum([rep['used_sub'] for rep in pu_reports]),
            'used_mul': sum([rep['used_mul'] for rep in pu_reports]),
            'used_div': sum([rep['used_div'] for rep in pu_reports]),
            'used_cpr': sum([rep['used_cpr'] for rep in pu_reports]),
            'length_avg': sum([rep['length_avg'] for rep in pu_reports]),
            'seq_tacts': seq_tacts,
            'par_tacts': par_tacts,
            'acceleration_coeff': seq_tacts / par_tacts,
            'efficency': efficency,
            'length': par_tacts,
            'procs_amount': self._procs_amount
        }
    

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
        self._matrix_e = matrices['E']
        self._matrix_g = matrices['G']

        # Assigning computation units
        self._pu = ComplexCU(config)
        
        self._matrix_c = None
        self._matrix_f = None
        self._matrix_d = None
    
    def compute_d(self):        
        a = list(
            chain.from_iterable(
                [row  * self._q for row in self._matrix_a]
            )
        )
        b = list(
            chain.from_iterable(zip(*self._matrix_b))
        ) * self._p       
        conj = self._pu.conj(a, b)       
        d = self._pu.cdisj(
            [conj[k:(k + self._m)] for k in range(0, len(conj), self._m)]
        )        
        self._matrix_d = np.array(d).reshape((self._p, self._q)).tolist()        
    
    def compute_f(self):        
        a = list(
            chain.from_iterable(
                [row  * self._q for row in self._matrix_a]
            )
        )
        b = list(
            chain.from_iterable(zip(*self._matrix_b))
        ) * self._p        
        e = self._matrix_e * (self._p * self._q)

        # Computing terms because expression is big
        # Left part
        t1: Iterable[Number] = self._pu.impl(a, b)
        t2: Iterable[Number] = self._pu.sub(self._pu.mul([2] * len(e), e), [1] * len(e))
        t3: Iterable[Number] = self._pu.mul(t1, t2)
        t_left = self._pu.mul(t3, e)

        # Right part
        t1: Iterable[Number] = self._pu.impl(b, a)
        t2: Iterable[Number] = self._pu.impl(a, b)
        t3: Iterable[Number] = self._pu.sub(
            self._pu.mul([4] * len(t2), t2), [2] * len(t2)
        )
        t4: Iterable[Number] = self._pu.mul(t3, e)
        t5: Iterable[Number] = self._pu.add([1] * len(t4), t4)
        t6: Iterable[Number] = self._pu.sub([1] * len(e), e)
        t7: Iterable[Number] = self._pu.mul(t1, t5)
        t_right: Iterable[Number] = self._pu.mul(t7, t6)

        # Final term
        f = self._pu.add(t_left, t_right)
        self._matrix_f = np.array(f).reshape(self._p, self._q, self._m).tolist()        
    
    def compute_c(self):
        if self._matrix_f is None:
            self.compute_f()
        if self._matrix_d is None:
            self.compute_d()
        
        f: Iterable[Iterable[Number]] = list(chain.from_iterable(self._matrix_f))
        d: Iterable[Number] = list(chain.from_iterable(self._matrix_d))
        g: Iterable[Number] = list(chain.from_iterable(self._matrix_g))

        # Computing terms because expression is big
        # Left part        
        t1: Iterable[Number] = self._pu.cconj(f)        
        t2: Iterable[Number] = self._pu.sub(self._pu.mul([3] * len(g), g), [2] * len(g))
        t3: Iterable[Number] = self._pu.mul(t1, t2)
        t_left: Iterable[Number] = self._pu.sub(t3, g)

        # Right part
        t1: Iterable[Number] = self._pu.sub([1] * len(g), g)
        t2: Iterable[Number] = self._pu.cconj(f)
        t3: Iterable[Number] = self._pu.disj(t2, d)
        t4: Iterable[Number] = self._pu.mul([4] * len(t3), t3)
        t5: Iterable[Number] = self._pu.mul([3] * len(d), d)
        t6: Iterable[Number] = self._pu.sub(t4, t5)
        t7: Iterable[Number] = self._pu.mul(t6, g)
        t8: Iterable[Number] = self._pu.add(d, t7)
        t_right: Iterable[Number] = self._pu.mul(t1, t8)

        # Final term
        c = self._pu.add(t_left, t_right)
        self._matrix_c = np.array(c).reshape((self._p, self._q)).tolist()        
    
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
            self._matrix_e,
            self._matrix_g,
            self._matrix_d,
            self._matrix_f
        )

    @property
    def matrices(self) -> dict:
        if self._matrix_c is None:
            self.compute_c()        
        return {
                'matrix_A': self._matrix_a,
                'matrix_B': self._matrix_b,
                'matrix_E': self._matrix_e,
                'matrix_G': self._matrix_g,
                'matrix_D': self._matrix_d,
                'matrix_F': self._matrix_f,
                'matrix_C': self._matrix_c,                
            }
    
    @property
    def report(self):
        matrices = self.matrices
        prog_rank = sum([prod(MatrixUtility.shape(matrix))
                         for matrix in self.data_matrices])
        report = self._pu.report
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


