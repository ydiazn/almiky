'''Embedding methods based on dither modulation

From: Chen, B., & Wornell, G. W. (2001). Quantization index modulation:
A class of provably good methods for digital watermarking and
information embedding. IEEE Transactions on
Information theory, 47(4), 1423-1443.
'''

import random

import numpy as np

from almiky.embedding import Embedder


class RandomDitherValue:
    '''
    Random binary dither value

    Generate a random value depending of quantization step
    in range [-∆/2, ∆/2] using uniform distribution.

    Args:
        step (float): quantization step

    Returns: dither value.

    Example:
        >>> from almiky.embedding.qim import RandomDitherValue
        >>> x = RandomDitherValue(12)
        >>> x
        -1.5809799177776949
    '''

    def __new__(cls, step):
        '''Initialize x, see help(type(x))'''

        value = random.uniform(-step / 2, step / 2)
        return value


class BinaryDM(Embedder):
    '''
    Non coded binary Dither Modulation

    - Binary: Rm = 1, m = {0, 1}
    - Uncoded case: zi = bi, ku/kc = 1
    - Host signal: x ∈ R

    Args:
        quantizer (integer): scalar quantizer
        d0 (float): dither value for m = 0, default to None

    Constraints:
        ∆: quantization step
        d0: must be between in [-∆/2, ∆/2]

    Construction:
        q = UniformQuantizer(step=10)
        BinaryDM(q, d0=2.25) => new dither modulation embedder with
        ∆=10, d0=2.25 and d1=-2.25
    '''

    def __init__(self, quantizer, d0):
        '''Initialize x; see help(type(x)) for details'''

        self.quantizer = quantizer

        # Set dither values
        d1 = d0 - np.sign(d0 or 1) * quantizer.step / 2
        self.dither_sequence = np.array([d0, d1])

    def dither(self, m):
        '''
        Return indexed dither value

        Args:
            m (integer): index (value 0 or 1)

        Returns:
            float: dither sequence
        '''

        return self.dither_sequence[m]

    def embed(self, amplitude, bit):
        '''
        Embed a bit and return the new amplitude.

        An indexed quantization is used.
        Bit to embed is used as index.

        Args:
            amplitude (float): amplitude of signal
            bit (int): bit to embed (value 0 or 1)

        Returns:
            float: new amplitude
        '''

        bit = int(bit)
        if bit not in (0, 1):
            raise ValueError('Embedding an invalid bit')

        return (
            self.quantizer(amplitude + self.dither(bit)) -
            self.dither(bit))

    def extract(self, amplitude):
        '''
        Extract a bit from signal. Return bit extracted.

        Args:
            amplitude (float): amplitude of signal

        Returns:
            int: watermark bit extrated (value 0 or 1)
        '''

        distances = [
            abs(self.embed(amplitude, bit) - amplitude)
            for bit in (0, 1)
        ]

        return np.argmin(distances)
