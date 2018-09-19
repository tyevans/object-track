import cv2
import numpy as np


class Mask(object):

    def __init__(self, mask, one_based=True):
        self.mask = mask
        self.one_based = one_based

    def as_transparency(self, image_np):
        b, g, r = cv2.split(image_np)
        if self.one_based:
            mask = self.mask * 255
        else:
            mask = self.mask
        return np.dstack([b, g, r, mask])

    def apply(self, image_np):
        if self.one_based:
            mask = self.mask * 255
        else:
            mask = self.mask
        return cv2.bitwise_and(image_np, image_np, mask=mask)

    def draw(self, image_np, color):
        for s_row, m_row in zip(image_np, self.mask):
            for s_pixel, m_pixel in zip(s_row, m_row):
                if m_pixel == 1:
                    s_pixel[...] = np.average([s_pixel, color], axis=0)
