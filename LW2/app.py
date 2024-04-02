from src.matrix_calculations import MatrixPU
from random import randint


matrixpu = MatrixPU({'q': randint(1, 10), 'p': randint(1, 10), 'm': randint(1, 10),
                     'ADD_TIME': randint(1, 10),
                     'SUB_TIME': randint(1, 10),
                     'MUL_TIME': randint(1, 10),
                     'DIV_TIME': randint(1, 10),
                     'CPR_TIME': randint(1, 10),
                     'PROCS_ELEMS': randint(1, 10)})
print(matrixpu.report)
