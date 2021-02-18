'''Quantization index modulation'''

from abc import ABC, abstractmethod


class Dither(ABC):
    '''
    Interface for indexed dither callback

    Args:
        index (integer): dither index

    Returns:
        dither value

    '''

    @abstractmethod
    def __call__(index):
        '''See help(type(instance))'''
