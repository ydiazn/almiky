from unittest import TestCase

from almiky.embedding.qim import Dither


class DitherInterfaceTest(TestCase):
    '''Test Dither interface'''

    def test(self):
        with self.assertRaises(TypeError):
            Dither()
