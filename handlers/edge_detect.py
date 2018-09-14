import cv2
import numpy as np

from detect.annotation import Mask
from handlers import ImageHandler


class CannyEdgeDetector(ImageHandler):

    def apply(self, image_np):
        edges = cv2.Canny(cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY), 100, 200)
        ret, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY_INV)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


class CannyEdgeEnhancer(ImageHandler):

    def apply(self, image_np):
        edges = cv2.Canny(cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY), 100, 200)
        ret, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY_INV)
        print(thresh.shape)
        mask = Mask(np.squeeze(thresh))
        return mask.apply(image_np)


class Laplacian(ImageHandler):

    def apply(self, image_np):
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        ret, thresh = cv2.threshold(edges, 8, 255, cv2.THRESH_BINARY_INV)
        opened = cv2.morphologyEx(thresh.astype(np.uint8), cv2.MORPH_OPEN, (5, 5))
        return cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)


class LaplacianEnhancer(ImageHandler):

    def apply(self, image_np):
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        ret, thresh = cv2.threshold(edges, 8, 255, cv2.THRESH_BINARY_INV)
        opened = cv2.morphologyEx(thresh.astype(np.uint8), cv2.MORPH_OPEN, (3, 3))
        mask = Mask(opened)
        return mask.apply(image_np)


class SobelX(ImageHandler):

    def apply(self, image_np):
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        # ret, thresh = cv2.threshold(edges, 8, 255, cv2.THRESH_BINARY_INV)
        # opened = cv2.morphologyEx(thresh.astype(np.uint8), cv2.MORPH_OPEN, (5, 5))
        return cv2.cvtColor(edges.astype(np.uint8), cv2.COLOR_GRAY2BGR)


class SobelXEnhancer(ImageHandler):

    def apply(self, image_np):
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        ret, thresh = cv2.threshold(edges, 8, 255, cv2.THRESH_BINARY_INV)
        opened = cv2.morphologyEx(thresh.astype(np.uint8), cv2.MORPH_OPEN, (3, 3))
        mask = Mask(opened)
        return mask.apply(image_np)

class SobelY(ImageHandler):

    def apply(self, image_np):
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        edges = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        # ret, thresh = cv2.threshold(edges, 8, 255, cv2.THRESH_BINARY_INV)
        # opened = cv2.morphologyEx(thresh.astype(np.uint8), cv2.MORPH_OPEN, (5, 5))
        return cv2.cvtColor(edges.astype(np.uint8), cv2.COLOR_GRAY2BGR)


class SobelYEnhancer(ImageHandler):

    def apply(self, image_np):
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        edges = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        ret, thresh = cv2.threshold(edges, 8, 255, cv2.THRESH_BINARY_INV)
        opened = cv2.morphologyEx(thresh.astype(np.uint8), cv2.MORPH_OPEN, (3, 3))
        mask = Mask(opened)
        return mask.apply(image_np)