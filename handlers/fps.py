import cv2

from handlers import ImageHandler


class FPSCounter(ImageHandler):

    def __init__(self):
        self.timer = cv2.getTickCount()

    def apply_first(self, image_np):
        return self.apply(image_np)

    def apply(self, image_np):
        timer = cv2.getTickCount()
        fps = int(cv2.getTickFrequency() / (timer - self.timer))

        font = cv2.FONT_HERSHEY_PLAIN
        fontScale = 1
        lineType = 2
        cv2.putText(image_np, "{} FPS".format(fps), (10, 40), font,
                    fontScale,
                    (0, 0, 255),
                    lineType)
        self.timer = timer
        return image_np
