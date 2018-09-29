import numpy as np

import cv2

from annotate.mask import Mask
from image_handlers import ImageHandler
from image_handlers.color import BilateralFilter, Quantize


class Cartoonify(ImageHandler):

    def __init__(self, quant_clusters=4):
        self.bilateral_filter = BilateralFilter()
        self.quant = Quantize(quant_clusters)

    def apply(self, image_np):
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        thresh = cv2.threshold(edges, 8, 255, cv2.THRESH_BINARY_INV)[1].astype(np.uint8)
        mask = Mask(thresh)
        return mask.apply(
            self.quant.apply(
                self.bilateral_filter.apply(image_np)
            )
        )
