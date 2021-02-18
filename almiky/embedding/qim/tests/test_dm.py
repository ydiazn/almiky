'''Test for methods based on quantization index modulation (QIM)'''

import random
from unittest import TestCase
from unittest.mock import call, MagicMock, Mock

from almiky.embedding.qim import dm
from almiky.quantization.scalar import UniformQuantizer


class BinaryDMDitherTest(TestCase):
    '''Test BinaryDM dither function d(m)'''

    def test_d0_out_of_range(self):
        '''Test dither function for a positive d0'''

        with self.assertRaises(ValueError):
            dm.BinaryDither(step=12, d0=6.5)

        with self.assertRaises(ValueError):
            dm.BinaryDither(step=12, d0=-6.5)

        # Value on limit on range
        dm.BinaryDither(step=12, d0=-6)
        dm.BinaryDither(step=12, d0=6)

    def test_positive_d0(self):
        '''Test dither function for a positive d0'''

        d = dm.BinaryDither(step=12, d0=3)

        self.assertEqual(d(0), 3)
        self.assertEqual(d(1), -3)

    def test_negative_d0(self):
        '''Test dither function for a positive d0'''

        d = dm.BinaryDither(step=12, d0=-2)

        self.assertEqual(d(0), -2)
        self.assertEqual(d(1), 4)

    def test_zero_d0(self):
        '''Test dither function for a positive d0'''

        d = dm.BinaryDither(step=12, d0=0)

        self.assertEqual(d(0), 0)
        self.assertEqual(d(1), -6)


class BinaryDMEmbedTest(TestCase):

    def test_non_binary_data(self):
        quantizer = MagicMock()
        dither = MagicMock()
        emb = dm.BinaryDM(quantizer, dither)

        with self.assertRaises(ValueError):
            emb.embed(10, 5)

    def test_embedding_cero(self):
        quantizer = MagicMock(return_value=30)
        # positve d(m)
        dither = MagicMock(return_value=4)
        emb = dm.BinaryDM(quantizer, dither)

        self.assertEqual(emb.embed(25, 0), 26)
        quantizer.assert_called_with(29)
        self.assertListEqual(emb.dither.call_args_list, [call(0), call(0)])

        # negative d(m)
        dither = MagicMock(return_value=-4)
        emb = dm.BinaryDM(quantizer, dither)

        self.assertEqual(emb.embed(25, 0), 34)
        quantizer.assert_called_with(21)
        self.assertListEqual(emb.dither.call_args_list, [call(0), call(0)])

    def test_embedding_one(self):
        quantizer = MagicMock(return_value=-30)

        # positive d(m)
        dither = MagicMock(return_value=4)
        emb = dm.BinaryDM(quantizer, dither)

        self.assertEqual(emb.embed(-25, 1), -34)
        quantizer.assert_called_with(-21)
        self.assertListEqual(emb.dither.call_args_list, [call(1), call(1)])

        # negative d(m)
        dither = MagicMock(return_value=-4)
        emb = dm.BinaryDM(quantizer, dither)

        self.assertEqual(emb.embed(-25, 1), -26)
        quantizer.assert_called_with(-29)
        self.assertListEqual(emb.dither.call_args_list, [call(1), call(1)])


class BinaryDMExtractTest(TestCase):

    def test_zero_embedded(self):
        quantizer = MagicMock()
        dither = MagicMock()
        emb = dm.BinaryDM(quantizer, dither)

        # emb.embed(x, 0) => 12
        # emb.embed(x, 1) => 5
        emb.embed = Mock(side_effect=[12, 5])

        self.assertEqual(emb.extract(10), 0)
        self.assertListEqual(
            emb.embed.call_args_list, [call(10, 0), call(10, 1)])

        # Equal distance
        emb.embed = Mock(side_effect=[2, 2])
        self.assertEqual(emb.extract(10), 0)
        self.assertListEqual(
            emb.embed.call_args_list, [call(10, 0), call(10, 1)])

    def test_one_embedded(self):
        quantizer = MagicMock()
        dither = MagicMock()
        emb = dm.BinaryDM(quantizer, dither)

        # emb.embed(x, 0) => 20
        # emb.embed(x, 1) => 5
        emb.embed = Mock(side_effect=[20, 5])

        self.assertEqual(emb.extract(10), 1)
        self.assertListEqual(
            emb.embed.call_args_list, [call(10, 0), call(10, 1)])

        # Equal distance
        emb.embed = Mock(side_effect=[2, 2])
        self.assertEqual(emb.extract(10), 0)
        self.assertListEqual(
            emb.embed.call_args_list, [call(10, 0), call(10, 1)])


class BinaryDMEmbeddExtractTest(TestCase):
    '''Test with Scalar Uniform Quatizer'''

    def test_without_noise(self):
        step = 12
        d0 = random.uniform(-6, 6)
        x = random.random() * 100

        quantizer = UniformQuantizer(step)
        dither = dm.BinaryDither(step, d0)
        emb = dm.BinaryDM(quantizer, dither)

        # embed a zero
        self.assertEqual(emb.extract(emb.embed(x, 0)), 0)
        # embed a one
        self.assertEqual(emb.extract(emb.embed(x, 1)), 1)

    def test_with_tolerable_noise(self):

        def add_noise(x, step):
            '''
            Add tolerable noise. Return noisy signal

            Arguments
            x -- amplitude of signal
            step -- quantization step
            '''
            noise = random.random() * step / 4
            return x + random.choice([-1, 1]) * noise

        step = 6
        d0 = random.uniform(-3, 3)
        x = random.random() * 100

        quantizer = UniformQuantizer(step)
        dither = dm.BinaryDither(step, d0)
        emb = dm.BinaryDM(quantizer, dither)

        # embed a zero
        sm = emb.embed(x, 0)
        y = add_noise(sm, step)
        self.assertEqual(emb.extract(y), 0)
        # embed a one
        sm = emb.embed(x, 1)
        y = add_noise(sm, step)
        self.assertEqual(emb.extract(y), 1)


class RandomDitherValueTest(TestCase):

    def test_value(self):
        value = dm.RandomDitherValue(12)

        self.assertLessEqual(abs(value), 6)
