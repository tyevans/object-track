import cv2
import numpy as np

from annotate.mask import Mask
from image_handlers import ImageHandler


class BackgroundSubtractor(ImageHandler):

    def __init__(self):
        """ Removes background (stationary) elements from an image, accumulating information across multiple images.
        """
        self.annotations = None
        self.subtractor = cv2.createBackgroundSubtractorKNN(history=1000)
        self.kernel = np.ones((3, 3), np.uint8)

    def apply(self, image_np):
        self.subtractor.apply(image_np)
        motion_mask = self.subtractor.apply(image_np)

        motion_mask = cv2.dilate(motion_mask, self.kernel, iterations=20)
        motion_mask = cv2.erode(motion_mask, self.kernel, iterations=20)

        self.motion_mask = Mask(motion_mask)
        return self.motion_mask.apply(image_np)

class BackgroundExtractor(ImageHandler):

    def __init__(self):
        """ Removes background (stationary) elements from an image, accumulating information across multiple images.
        """
        self.annotations = None
        self.subtractor = cv2.createBackgroundSubtractorKNN(history=1000)
        self.kernel = np.ones((3, 3), np.uint8)

    def apply(self, image_np):
        self.subtractor.apply(image_np)
        motion_mask = self.subtractor.apply(image_np)

        motion_mask = cv2.dilate(motion_mask, self.kernel, iterations=20)
        motion_mask = cv2.erode(motion_mask, self.kernel, iterations=20)

        self.motion_mask = Mask(np.logical_not(motion_mask))
        return self.motion_mask.apply(image_np)

class FrameAverager(ImageHandler):
    def __init__(self, width, height, history=20):
        self.history = history
        self.frame_index = 0
        self.frame_stack = np.zeros([history, height, width, 3])

    def apply(self, image_np):
        self.frame_stack[self.frame_index] = image_np
        self.frame_index = (self.frame_index + 1) % self.history
        avg = np.average(self.frame_stack, axis=0)
        return avg.astype(np.uint8)