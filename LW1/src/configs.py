# Метода решения задач в интеллектуальных системах
# Лабораторная работа №1 Вариант 7
# Авторы: Заломов Р.А., Готин И.А.
# Дата: 20.02.24
# Данный файл обеспечивает чтение конфигурационного файла,
# содердащего параметры работы арифметического конвейера


import os
import json


class Configs:

    def __init__(self) -> None:
        with open(os.path.join(os.path.dirname(__file__), 'configs.json')) as file:
            self._configs = json.load(file)
        
    @property
    def pipeline_levels_amount(self):
        return self._configs['PIPELINE_LEVELS_AMOUNT']
    
    @property
    def input_numbers_digit_amount(self):
        return self._configs['INPUT_NUMBERS_DIGIT_AMOUNT']
