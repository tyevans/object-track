import cv2

from annotate.mask import Mask
from image_handlers import ImageHandler


class Threshold(ImageHandler):

    def __init__(self, mode=cv2.THRESH_BINARY):
        self.mode = mode

    def apply(self, image_np):
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, self.mode)
        mask = Mask(thresh)
        return mask.apply(image_np)
