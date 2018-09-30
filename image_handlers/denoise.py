import cv2

from image_handlers import ImageHandler


class Denoise(ImageHandler):

    def apply(self, image_np):
        return cv2.fastNlMeansDenoisingColored(image_np,None,10,10,7,21)