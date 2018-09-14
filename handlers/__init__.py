class ImageHandler(object):
    """ Base class for a process that transforms an image

    Used to build pipelines for processing images
    """

    def apply_first(self, image_np):
        """
        Called on the first image/frame when a pipeline is started

        :param image_np: (np.array) cv2 image array (gbr)
        :return: (np.array) transformed image
        """
        return self.apply(image_np)

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
