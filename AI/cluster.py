#!/usr/bin/env python3

"""
Clustering functions for semantic segmentation
"""

# __author__ = "Nektarios Aristeidis Tafanidis"
# __email__ = "Nektariostaf@gmail.com"
# __website__="https://github.com/NekTaf/"

import numpy as np
import cv2 as cv


class Clustering(object):
    def __init__(self, img, clusters):
        """
        :param img: input image
        :param clusters: number of clusters
        """
        self.img = img
        self.clusters = clusters

    def KM(self, iter=3):

        # K-Means number of iterations (3 preassigned value)
        self.iter = iter

        Z = self.img.reshape((-1, 3))
        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, iter, 1.0)

        ret, label, center = cv.kmeans(Z, self.clusters, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        imageKM = res.reshape((self.img.shape))

        return imageKM
