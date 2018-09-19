import cv2
import numpy as np

from image_handlers import ImageHandler


class HistogramNormalize(ImageHandler):

    def apply(self, image_np):
        hist, bins = np.histogram(image_np.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        return cdf[image_np]


class CLAHE(ImageHandler):

    def apply(self, image_np):
        b, g, r = cv2.split(image_np)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        b = clahe.apply(b)
        g = clahe.apply(g)
        r = clahe.apply(r)
        return np.dstack([b, g, r])


class CLAHE_GRAY(ImageHandler):

    def apply(self, image_np):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        grey = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(clahe.apply(grey), cv2.COLOR_GRAY2BGR)
