import cv2


class ImageHandler(object):
    """ Base class for a process that transforms an image

    Used to build pipelines for processing images
    """

    def apply(self, image_np):
        """
        Called on all subsequent frames

        :param image_np: (np.array) cv2 image array (gbr)
        :return: (np.array) transformed image
        """
        return image_np

    def close(self):
        """
        Called when processing is complete
        """
        pass


class Pipeline(ImageHandler):

    def __init__(self, handlers):
        self.handlers = handlers

    def apply(self, image_np):
        image_np_copy = image_np.copy()
        for handler in self.handlers:
            image_np_copy = handler.apply(image_np_copy)
        return image_np

    def close(self):
        for handler in self.handlers:
            handler.close()


class VideoDisplay(ImageHandler):

    def __init__(self, window_name):
        self.window_name = window_name

    def apply(self, image_np):
        cv2.imshow(self.window_name, image_np)
        return image_np
