'''Scalar quantization module'''

from almiky.quantization import Quantizer


class UniformQuantizer(Quantizer):
    '''
    Scalar uniform quantizer

    Arguments:
        step: quantization step size (∆)

    UniformQuantizer(10) => new uniform quantizer with ∆=10

    Example:
        >>> q = UniformQuatizer(10)
        >>> q(12)
        10
    '''

    def __init__(self, step):
        '''Initialize x; see help(type(x)) for details'''
        self.step = step

    def __call__(self, amplitude):
        '''
        Quantize a signal

        Args:
            amplitude (float): amplitude of signal

        Returns:
            quantizer value
        '''
        return self.step * round(amplitude / self.step)
