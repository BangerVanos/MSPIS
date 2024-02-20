from typing import Literal
from dataclasses import dataclass, field


class BinaryNumber:
    def __init__(self, number: int | list[Literal[0, 1]], bit_amount: int) -> None:
        if isinstance(number, int):
            if number > (2**bit_amount - 1):
                number = 2**bit_amount - 1
            self._number: list[Literal[0, 1]] = list(map(int,
                                                     list(format(int(abs(number)),
                                                                 f'0{bit_amount}b'))))
        else:
            if len(number) > bit_amount:
                self._number = number[-bit_amount:]
            else:
                self._number = [0] * (bit_amount - len(number)) + number    
        self._bit_amount = bit_amount       
    
    @property
    def vector(self) -> list[Literal[0, 1]]:
        return self._number
    
    @property
    def bit_amount(self):
        return self._bit_amount

    @property
    def decimal(self) -> int:
        return sum([2**i*self._number[-(i+1)]
                    for i in range(len(self._number))])
    
    @property
    def binary(self) -> str:
        return ''.join(list(map(str, self._number)))

    def increase_bit_amount(self, new_bit_amount: int):
        if new_bit_amount > self._bit_amount:
            self._number = [0] * (new_bit_amount - self._bit_amount) \
                            + self._number
            self._bit_amount = new_bit_amount
        return self

    def __lshift__(self, other):
        if not isinstance(other, int) or other < 0:
            return
        if other >= self._bit_amount:
            return_vector = [0] * self._bit_amount, self._bit_amount
        else:
            return_vector = self._number[other:] + [0] * other
        return BinaryNumber(return_vector, self._bit_amount)
    
    def __add__(self, other):
        if not isinstance(other, BinaryNumber):
            return None
        self_vector = [0] * (other.bit_amount - self.bit_amount) + self._number \
                       if other.bit_amount > self._bit_amount else self._number
        other_vector = [0] * (self.bit_amount - other.bit_amount) + other.vector\
                        if self._bit_amount > other.bit_amount else other.vector
        result_vector = []
        result_bit_amount = self._bit_amount if self._bit_amount > other.bit_amount \
                                                else other.bit_amount
        carry = 0
        for i in range(-1, -(len(self_vector) + 1), -1):
            digit_self = self_vector[i]
            digit_other = other_vector[i]
            result_digit = digit_self ^ digit_other ^ carry
            carry = digit_self & digit_other | (digit_self ^ digit_other) & carry
            result_vector.append(result_digit)
        return BinaryNumber(result_vector[::-1], result_bit_amount)

    def __and__(self, other):
        if not isinstance(other, BinaryNumber):
            return None
        self_vector = [0] * (other.bit_amount - self.bit_amount) + self._number \
                       if other.bit_amount > self._bit_amount else self._number
        other_vector = [0] * (self.bit_amount - other.bit_amount) + other.vector\
                        if self._bit_amount > other.bit_amount else other.vector
        result_vector = []
        result_bit_amount = self._bit_amount if self._bit_amount > other.bit_amount \
                                                else other.bit_amount        
        for i in range(-1, -(len(self_vector) + 1), -1):
            digit_self = self_vector[i]
            digit_other = other_vector[i]
            result_digit = digit_self & digit_other            
            result_vector.append(result_digit)
        return BinaryNumber(result_vector[::-1], result_bit_amount)

    def __repr__(self) -> str:
        return f'BinaryNumber({''.join(list(map(str, self._number)))} = {self.decimal})'

    def __str__(self) -> str:
        return self.__repr__()             


class BinaryMultiplicator:
    def __init__(self, multiplicand: BinaryNumber, multiplier: BinaryNumber) -> None:
        self._needed_shifts = multiplicand.bit_amount

        self._bit_amount = 2*max(multiplicand.bit_amount, multiplier.bit_amount)
        self._multiplicand = multiplicand.increase_bit_amount(self._bit_amount)
        self._multiplier = multiplier.increase_bit_amount(self._bit_amount)
        
        self._partial_sum = BinaryNumber(0, self._bit_amount)
        self._partial_product = BinaryNumber(0, self._bit_amount)
        self._shifts_amount = 0

        self._is_done = False
    
    def make_step(self):
        if self._shifts_amount < self._needed_shifts:
            self._partial_product = (BinaryNumber(0, self._bit_amount),
                                     self._multiplicand)[self._multiplier.vector[-(self._shifts_amount + 1)]]            
            self._partial_sum = self._partial_sum + self._partial_product
            self._shifts_amount += 1
            self._multiplicand = self._multiplicand << 1                        
        else:
            self._is_done = True
         
    @property
    def is_done(self) -> bool:
        return self._is_done
    
    @property
    def result(self):
        return self._partial_sum if self._is_done else None

    @property
    def partial_sum(self):
        return self._partial_sum
    
    @property
    def partial_product(self):
        return self._partial_product


class ArithmeticPipelineLevel:
    def __init__(self) -> None:
        self._is_busy = False
        self._operator: BinaryMultiplicator | None = None

        self._work_result: BinaryNumber | None = None
    
    @property
    def operator(self) -> BinaryMultiplicator:
        return self._operator
    
    @operator.setter
    def operator(self, operator: BinaryMultiplicator | None):
        self._operator = operator
        self._is_busy = True if isinstance(operator, BinaryMultiplicator) else False
    
    def free(self):
        self._operator = None
        self._is_busy = False
        self._work_result = None
    
    def make_step(self):
        if self._operator:
            if not self._operator.result:
                self._operator.make_step()
            else:
                self._work_result = self._operator.result                
    
    @property
    def is_busy(self):
        return self._is_busy
    
    @property
    def work_result(self) -> BinaryNumber | None:
        return self._work_result

    def __str__(self) -> str:
        if not self._operator:
            partial_sum = '---'
            partial_product = '---'
        else:
            partial_sum = self._operator.partial_sum.binary
            partial_product = self._operator.partial_product.binary
        return f'Partial sum: {partial_sum} | Partial product: {partial_product}'


class ArithmeticPipeline:
    def __init__(self, vector_1: list[int], vector_2: list[int], levels_amount: int,
                 number_bit_amount: int) -> None:
        self._vector_1 = vector_1[::-1]
        self._vector_2 = vector_2[::-1]
        self._result_vector: list[int] = []

        self._levels_amount = levels_amount
        self._levels = {level: ArithmeticPipelineLevel() for level in range(self._levels_amount)}

        self._pair_indexes = {level: -1 for level in range(self._levels_amount)}
        self._last_pair_inserted_index = -1

        self._busy = True
        self._tacts_done = 0

        self._number_bit_amount: int = number_bit_amount
    
    def tact(self) -> None:
        if (not any([level.is_busy for level in self._levels.values()]) 
              and not self._vector_1 and not self._vector_2):            
            self._become_free()
            return
        if (not self._levels[0].is_busy
            and self._vector_1 and self._vector_2):
            self._add_task()
        self._level_steps()
        if self._tacts_done and not self._levels[self._levels_amount - 1].is_busy:
            self._move_pipeline()                
        if self._levels[self._levels_amount - 1].work_result:
            self._save_level_result(self._levels_amount - 1)                                       
            self._free_level(self._levels_amount - 1)                                                                      
        self._tacts_done += 1    
    
    def _add_task(self) -> None:
        self._last_pair_inserted_index += 1
        nearest_level = self._levels[0]        
        multiplicand = BinaryNumber(self._vector_1.pop(), self._number_bit_amount)
        multiplier = BinaryNumber(self._vector_2.pop(), self._number_bit_amount)
        self._pair_indexes[0] = self._last_pair_inserted_index
        nearest_level.operator = BinaryMultiplicator(multiplicand, multiplier)

    def _level_steps(self) -> None:
        for pipeline_level in self._levels.values():
            if not pipeline_level.work_result:
                pipeline_level.make_step()        
    
    def _move_pipeline(self) -> None:
        first_busy_level = min([index for index, level in self._levels.items()
                                if level.is_busy])
        if first_busy_level == self._levels_amount - 1:
            return                    
        previous_operator = self._levels[first_busy_level].operator
        previous_pair_index = self._pair_indexes[first_busy_level]            
        self._free_level(first_busy_level)                            
                                                         
        for index in range(first_busy_level + 1, self._levels_amount):                                                   
                self._levels[index].operator, previous_operator = previous_operator, self._levels[index].operator
                self._pair_indexes[index], previous_pair_index = previous_pair_index, self._pair_indexes[index]                    
        
    def _save_level_result(self, index) -> None:
        free_level = self._levels[index]
        work_result = free_level.work_result.decimal
        self._result_vector.append(work_result)
    
    def _free_level(self, index) -> None:
        free_level = self._levels[index]          
        free_level.free()
        self._pair_indexes[index] = -1                     
    
    def _become_free(self):
        self._busy = False
    
    @property
    def is_busy(self):
        return self._busy
    
    @property
    def result(self) -> None | list[int]:
        return self._result_vector if not self._busy else None
    
    def __str__(self) -> str:        
        return (f'----------\n'
                f'Tact {self._tacts_done}\n'
                f'Numbers queue 1: {self._vector_1[::-1]}\n'
                f'Numbers queue 2: {self._vector_2[::-1]}\n'
                f'{'\n'.join([f'Level {i+1}\n'
                              f'Pair index: {'--' if (pair_index := self._pair_indexes[i] + 1) == 0 else pair_index}\n'
                              f'{level.__str__()}'
                              for i, level in self._levels.items()])}\n'
                f'Result: {self._result_vector}'
                )        