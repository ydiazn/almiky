'''Embedding methods based on dither modulation

From: Chen, B., & Wornell, G. W. (2001). Quantization index modulation:
A class of provably good methods for digital watermarking and
information embedding. IEEE Transactions on
Information theory, 47(4), 1423-1443.
'''

import random

import numpy as np

from almiky.embedding import Embedder
from almiky.embedding.qim import Dither


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


class BinaryDither(Dither):
    '''
    Binary indexed dither callback

    Args:
        index (integer): dither index

    Returns:
        dither value

    Constraints:
        ∆: quantization step
        d0: must be between in [-∆/2, ∆/2]
    '''

    def __init__(self, step, d0):
        '''Initialize x, see help(type(instance))'''

        # d0 must be in [-step / 2, step / 2]
        if abs(d0) > step / 2:
            raise ValueError

        self.step = step
        self.d0 = d0
        self.values = np.empty(2)
        self.cached = False

    def __call__(self, index):
        '''See help(type(instance))'''

        if not self.cached:
            d1 = self.d0 - np.sign(self.d0 or 1) * self.step / 2
            self.values = np.array([self.d0, d1])
            self.cached = True

        return self.values[index]


class BinaryDM(Embedder):
    '''
    Non coded binary Dither Modulation

    - Binary: Rm = 1, m = {0, 1}
    - Uncoded case: zi = bi, ku/kc = 1
    - Host signal: x ∈ R

    Args:
        quantizer (Quantizer): scalar quantizer
        dither (Dither): binary dither

    Example:
        Binary dither modulation with ∆=10, d0=-3 and d1=3

        >>> from almiky.quantization.scalar import UniformQuantizer
        >>> from almiky.embedding.qim import dm
        >>> q = UniformQuantizer(step=12)
        >>> d = dm.BinaryDither(step=12, d0=-3)
        >>> emb = dm.BinaryDM(q, d)
        >>> x = 30
        >>> y = emb.embed(x, 0)
        >>> y
        27.0
        >>> emb.extract(y)
        0
    '''

    def __init__(self, quantizer, dither):
        '''Initialize x; see help(type(x)) for details'''

        self.quantizer = quantizer
        self.dither = dither

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
