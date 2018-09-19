import cv2
import numpy as np


class Mask(object):

    def __init__(self, mask, one_based=True):
        self.width, self.height = mask.shape[:2]
        self.mask = mask
        self.one_based = one_based

    def as_transparency(self, image_np):
        height, width = image_np.shape[:2]
        output = np.zeros([height, width, 4], dtype=np.uint8)
        for o_row, s_row, m_row in zip(output, image_np, self.mask):
            for o_pixel, s_pixel, m_pixel in zip(o_row, s_row, m_row):
                if m_pixel == 1 or m_pixel == 255:
                    o_pixel[...] = np.array([s_pixel[0], s_pixel[1], s_pixel[2], 255], dtype=np.uint8)
        return output

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
