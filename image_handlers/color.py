import cv2
import numpy as np

from image_handlers import ImageHandler


class ChannelFilter(ImageHandler):

    def __init__(self, channel_no=0):
        self.channel_no = channel_no

    def apply(self, image_np):
        channel = image_np[:, :, self.channel_no]
        return cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)

