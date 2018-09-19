import cv2
import numpy as np

from annotate.mask import Mask
from image_handlers import ImageHandler


class BackgroundSubtractor(ImageHandler):

    def __init__(self):
        """ Removes background (stationary) elements from an image, accumulating information across multiple images.
        """
        self.annotations = None
        self.subtractor = cv2.createBackgroundSubtractorMOG2()
        self.kernel = np.ones((5, 5), np.uint8)

    def apply(self, image_np):
        self.subtractor.apply(image_np)
        motion_mask = self.subtractor.apply(image_np)

        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, self.kernel)
        motion_mask = cv2.dilate(motion_mask, self.kernel, iterations=1)

        self.motion_mask = Mask(motion_mask)
        return self.motion_mask.apply(image_np)
