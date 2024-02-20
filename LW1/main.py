from src.arithmetic_pipeline import BinaryNumber, BinaryMultiplicator, ArithmeticPipeline
from src.configs import Configs


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
    pipeline = ArithmeticPipeline([1, 3, 5, 10, 21, 34, 20], [7, 9, 11, 56, 7, 2, 10],
                                  configs.pipeline_levels_amount,
                                  configs.input_numbers_digit_amount)
    while pipeline.is_busy:
        print(pipeline)
        pipeline.tact()
    print(pipeline)


if __name__ == '__main__':
    test_addition()
    test_multiplication()
    test_pipeline()
