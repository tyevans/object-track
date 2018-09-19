import cv2

from image_handlers import ImageHandler


class AVIOutput(ImageHandler):

    def __init__(self, output_path, width=640, height=480, frame_rate=30.0):
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._output = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    def apply(self, image_np):
        self._output.write(image_np)
        return image_np
