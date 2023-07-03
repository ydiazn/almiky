'''
Imperceptibility performance metrics
'''

from math import log10
import numpy as np


def mse(cover_work, ws_work):
    '''
    Calculate an return Mean Square Error (MSE)
    between a cover work an wattermaked work (or stego work)
    '''

    if cover_work.shape != ws_work.shape:
        raise ValueError('Cover and watermak/stego work have different shape')

    dims_cover = cover_work.shape
    m = dims_cover[0]
    n = dims_cover[1]
    C, S = cover_work * 1.0, ws_work * 1.0
    diff_pow = (np.abs(C) - np.abs(S)) ** 2
    if len(dims_cover) == 2:
        return sum(sum(diff_pow)) / (m * n)
    elif len(dims_cover) == 3:
        return sum(sum(sum(diff_pow))) / (m * n * dims_cover[2])


def psnr(cover_array, stego_array, max=255):
    '''
    Peak Signal-to-Noise Ratio (PSNR)
    '''
    RMSE = mse(cover_array, stego_array) or 1 / cover_array.size
    return 10 * log10(max ** 2 / RMSE) if RMSE else 100


def uiqi(cover_array, stego_array, axis=(0, 1)):
    '''
    Universal Image Quality Index

    Arguments:
    cover_array -- array containing cover data
    stego_array -- array containing stego data
    axis -- axes along which the uiqi is computed.
        The default is (0, 1).

    Usage:
    uiqi([[2, 6], [3, 2]], [[6, 2], [8, 5]])
    '''

    if cover_array.shape != stego_array.shape:
        raise ValueError('Cover and watermak/stego work have different shape')

    Mc = np.mean(cover_array, axis)
    Ms = np.mean(stego_array, axis)
    Mc2 = np.emath.power(Mc, 2)
    Ms2 = np.emath.power(Ms, 2)
    Vc = np.var(cover_array, axis, ddof=1)
    Vs = np.var(stego_array, axis, ddof=1)
    size = np.prod([np.size(cover_array, axis) for axis in axis])
    Scs = np.sum((cover_array - Mc) * (stego_array - Ms), axis) / (size - 1)
    index = (4 * Scs * Mc * Ms) / ((Vc + Vs) * (Mc2 + Ms2))

    return np.mean(index)
