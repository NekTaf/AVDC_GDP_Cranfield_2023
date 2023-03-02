#!/usr/bin/env python3

"""
Clustering functions for semantic segmentation
"""


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

    def KM(self, iter=100):

        # K-Means number of iterations (3 preassigned value)
        self.iter = iter

        pixel_val = self.img.reshape((-1, 3))
        # convert to np.float32
        pixel_val = np.float32(pixel_val)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, iter, 1)

        _, label, center = cv.kmeans(pixel_val, self.clusters, None, criteria, 10, cv.KMEANS_PP_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        segmented_image = center[label.flatten()]
        segmented_image = segmented_image.reshape((self.img.shape))

        return segmented_image
