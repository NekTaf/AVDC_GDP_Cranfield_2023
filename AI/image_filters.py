
"""
Filters for image processing and feature extraction
"""

import math
import cv2
import numpy as np


class Filtering(object):

    def __init__(self, img):
        """
        :param img: input image
        """
        self.img = img

    # Gabor filter
    def GF(self, ksize=51, sigma=3, psi=0):

        """
        :param ksize: Kernel size
        :param sigma: standard deviation of the Gaussian function
        :param psi:  phase offset of the sinusoidal function
        """
        self.ksize=ksize
        self.ksize=sigma
        self.psi=psi

        # Image Size
        image_size = self.img.shape
        num_col = image_size[1]
        num_row = image_size[0]

        wavelengthMin = 4 / math.sqrt(2)
        wavelengthMax = math.hypot(num_row, num_col)

        n = math.floor(math.log2(wavelengthMax / wavelengthMin))

        wavelength_array = wavelengthMin * 2 ** (np.arange(n - 1))
        orientation_array = np.arange(0, (181 - 45), 45)

        for i in range(len(orientation_array)):
            # Create the Gabor kernel
            for i2 in range(len(wavelength_array)):
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, wavelength_array[i2], orientation_array[i], 0.5,psi)
                # Apply the Gabor filter to the image
                filtered = cv2.filter2D(self.img, cv2.CV_32F, kernel)
                filtered = cv2.add(filtered, filtered)

        return filtered