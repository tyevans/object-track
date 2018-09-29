import cv2
import numpy as np


class Mask(object):

    def __init__(self, mask):
        self.mask = mask

    def as_transparency(self, image_np):
        b, g, r = cv2.split(image_np)
        return np.dstack([b, g, r, self.mask * 255])

    def apply(self, image_np):
        mask = self.mask * 255
        return cv2.bitwise_and(image_np, image_np, mask=mask)

    def draw(self, image_np, color):
        mask = image_np.copy()
        mask[self.mask == 1] = color
        image_np[...] = cv2.addWeighted(image_np, 0.5, mask, 0.5, 0)

    def overlay(self, bg, fg):
        fg[self.mask != 1] = (0, 0, 0)
        bg[self.mask == 1] = (0, 0, 0)
        bg[...] = cv2.add(bg, fg)

