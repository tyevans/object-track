from __future__ import division

import cv2

from handlers import ImageHandler


class AVIOutput(ImageHandler):

    def __init__(self, output_path, width=640, height=480, frame_rate=15.0):
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._output = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    def apply_first(self, image_np):
        return self.apply(image_np)

    def apply(self, image_np):
        self._output.write(image_np)
        return image_np

    def close(self):
        self._output.release()
