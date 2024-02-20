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
