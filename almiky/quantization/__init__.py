'''Quantization package'''

from abc import ABC, abstractmethod


class Quantizer(ABC):
    ''''
    Interface for a signal quantizer callback

    Args:
        step (float): quatization step
    '''

    @abstractmethod
    def __call__(amplitude):
        '''
        Quantize a signal

        Args:
            amplitude (float): amplitude of signal

        Returns:
            quantizer value
        '''
