'''Test for methods based on quantization index modulation (QIM)'''


from unittest import TestCase

from almiky.embedding.dpc import qim


class BitnaryQIMTest(TestCase):
    '''Test for binary QIM method'''

    def test_invalid_bit(self):
        '''Test embedding with an invalid bit'''
        embedder = qim.BinaryQuantizationIndexModulation(step=10)

        with self.assertRaises(ValueError):
            embedder.embed(150, 3)

    def test_embedding_one(self):
        '''Test embedding a one with arbitrary step size'''
        embedder = qim.BinaryQuantizationIndexModulation(step=10)

        modified = embedder.embed(150, 1)
        self.assertEqual(modified, 165)

        modified = embedder.embed(150, '1')
        self.assertEqual(modified, 165)

    def test_embedding_cero(self):
        '''Test embedding a cero with arbitrary step size'''
        embedder = qim.BinaryQuantizationIndexModulation(step=10)

        modified = embedder.embed(155, 0)
        self.assertEqual(modified, 155)

        # Testing that bit can be represented as str
        modified = embedder.embed(155, '0')
        self.assertEqual(modified, 155)

    def test_extraction(self):
        '''Test extraction with arbitrary step size'''
        embedder = qim.BinaryQuantizationIndexModulation(step=10)

        bit = embedder.extract(165)
        self.assertEqual(bit, 1)

        bit = embedder.extract(155)
        self.assertEqual(bit, 0)

    def test_extracting_with_big_step(self):
        '''Test the extracting with big step size compared with amplitude'''
        embedder = qim.BinaryQuantizationIndexModulation(step=40)
        bit = embedder.extract(20)
        self.assertEqual(bit, 1)

        bit = embedder.extract(-20)
        self.assertEqual(bit, 0)

    def test_inverse(self):
        '''Tested em.extract(em.insert(amplitude, bit) == bit'''
        embedder = qim.BinaryQuantizationIndexModulation(step=40)
        modified = embedder.embed(150, 1)
        bit = embedder.extract(modified)
        self.assertEqual(bit, 1)

        modified = embedder.embed(150, 0)
        bit = embedder.extract(modified)
        self.assertEqual(bit, 0)

        modified = embedder.embed(-150, 1)
        bit = embedder.extract(modified)
        self.assertEqual(bit, 1)

        modified = embedder.embed(-150, 0)
        bit = embedder.extract(modified)
        self.assertEqual(bit, 0)
