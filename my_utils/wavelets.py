import pywt
from skimage.transform import resize
import numpy as np


def mexh(signal: np.ndarray, max_width: int) -> np.ndarray:
    """
    :param signal: 1D ndarray of the timeseries signal to be processed
    :param max_width: max_width of wavepacket
    :return img: 2D ndarray
    """
    widths = range(1, max_width)
    img, _ = pywt.cwt(signal, widths, "mexh")
    img = resize(img, (128, 128))
    return img


def cmor(signal: np.ndarray, max_width: int) -> np.ndarray:
    """
    uses the complex-Morlet CWT
    :param signal: 1D ndarray of the timeseries signal to be processed
    :param max_width: max_width of wavepacket
    :return img: 2D ndarray
    """
    widths = range(1, max_width)
    img, _ = pywt.cwt(signal, widths, 'cmor1.5-1')
    img = abs(img)
    img = resize(img, (128, 128))
    return img
