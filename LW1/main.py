# Метода решения задач в интеллектуальных системах
# Лабораторная работа №1 Вариант 7
# Авторы: Заломов Р.А., Готин И.А.
# Дата: 20.02.24
# Данный файл содержит тестирование функций реализованной
# системы: сложение, умножение, работа арифметического конвейера 


from src.arithmetic_pipeline import BinaryNumber, BinaryMultiplicator, ArithmeticPipeline
from src.configs import Configs
from random import randint
from time import sleep


def test_addition():
    a = BinaryNumber(4, 6)
    b = BinaryNumber(8, 6)
    print((a + b).decimal)


def test_multiplication():
    a = BinaryNumber(5, 6)
    b = BinaryNumber(11, 6)
    mul = BinaryMultiplicator(a, b)
    while not mul._is_done:
        mul.make_step()
    print(mul.result.decimal)

def test_pipeline():
    configs = Configs()
    pipeline = ArithmeticPipeline([63], [63],
                                  configs.pipeline_levels_amount,
                                  configs.input_numbers_digit_amount)
    while pipeline.is_busy:
        print(pipeline)
        pipeline.tact()

if __name__ == '__main__':
    # test_addition()
    # test_multiplication()
    # test_pipeline()
    vector_1 = [randint(0, 63) for _ in range(15)]
    vector_2 = [randint(0, 63) for _ in range(15)]
    par_pipeline = ArithmeticPipeline(vector_1, vector_2, 6, 6)
    while par_pipeline.is_busy:
        par_pipeline.tact()
        # sleep(0.5)
        # print(par_pipeline.status)
    print(par_pipeline.status['tacts_done'])
