import cv2
import numpy as np
from sklearn.cluster import KMeans
from image_handlers import ImageHandler


class ChannelFilter(ImageHandler):

    def __init__(self, channel_no=0):
        self.channel_no = channel_no

    def apply(self, image_np):
        channel = image_np[:, :, self.channel_no]
        return cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)


class BilateralFilter(ImageHandler):
    kernel = np.ones([7, 7])

    def apply(self, image_np):
        for _ in range(8):
            image_np = cv2.bilateralFilter(image_np, 9, 20, 20)
        return image_np


class Quantize(ImageHandler):
    intialized = False

    def __init__(self, clusters):
        self.clt = KMeans(n_clusters=clusters)

    def apply(self, image_np):
        h, w = image_np.shape[:2]
        image = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)

        # reshape the image into a feature vector so that k-means
        # can be applied
        image = image.reshape((image.shape[0] * image.shape[1], 3))

        # apply k-means using the specified number of clusters and
        # then create the quantized image based on the predictions
        if self.intialized:
            labels = self.clt.predict(image)
            quant = self._clusters_centers[labels]
        else:
            self.intialized = True
            labels = self.clt.fit_predict(image)
            self._clusters_centers = self.clt.cluster_centers_.astype("uint8")
            quant = self._clusters_centers[labels]

        # reshape the feature vectors to images
        quant = quant.reshape((h, w, 3))

        return cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)