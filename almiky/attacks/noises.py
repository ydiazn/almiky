'''Noise attacks'''


import numpy as np


def salt_pepper_noise(image, density, max_value=255):
    """Applies Salt and Pepper noise to image.

    Args:
        image (numpy array): image data
        density (float):
            probability at the pixels are altered (value between 0 an 1)
        max_value (int, optional): maximun image values (default is 255)

    Returns:
        numpy array: noisy image
    """

    ones = np.ones(image.shape).astype('int')
    mask = (np.random.uniform(size=image.shape) < density).astype('int')
    imask = np.bitwise_xor(mask, ones)

    # Generation of salt & pepper
    noise = np.random.choice([0, 255], image.shape) * mask
    noisy = (image * imask) + noise

    return noisy


def gaussian_noise(image, percent_noise, max_value=255):
    '''Applies Gaussian noise to image.

    Args:
        image (numpy array): image data
        percent_noise (float): percent ratio of the standard deviation of
            the white Gaussian noise versus the signal for whole image
        max_value (int, optional): maximun image values (default is 255)

    Returns:
        numpy array: noisy image
    '''

    # Generation of gaussina noise with desired mu, sigma and density
    img_std = np.std(image)
    noise = np.random.normal(0, img_std * percent_noise, image.shape)

    noisy = image + noise
    # Ensuring valid noisy image data: value range and data type
    noisy = np.clip(noisy, 0, max_value).astype(image.dtype)

    return noisy
