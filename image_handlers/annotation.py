import cv2

import numpy as np

from annotate.video import TrackingVideoAnnotator, VideoAnnotator
from image_handlers import ImageHandler
from image_handlers.background import FrameAverager


class Annotator(ImageHandler):

    def __init__(self, graph: str, label_file: str, num_classes: int,
                 min_confidence: float = 0.5, color=None, sample_delay=30, width=640, height=480):
        self.color = color or (0, 255, 0)
        self.annotator = VideoAnnotator(width, height, graph, label_file, num_classes, min_confidence=min_confidence,
                                        sample_delay=sample_delay)

    def apply(self, image_np):
        self.annotator.step(image_np)
        for annotation in self.annotator.annotations:
            annotation.draw(image_np, self.color)
        return image_np

    def close(self):
        self.annotator.close()


class TrackingAnnotator(Annotator):

    def __init__(self, graph: str, label_file: str, num_classes: int,
                 min_confidence: float = 0.5, color=None, sample_delay=30, width=640, height=480):
        self.color = color or (0, 255, 0)
        self.annotator = TrackingVideoAnnotator(width, height, graph, label_file, num_classes, min_confidence=min_confidence,
                                                sample_delay=sample_delay)

class MaskingAnnotator(Annotator):
    anno_frame = None

    def apply(self, image_np):
        if self.anno_frame is None:
            self.anno_frame = image_np.copy()
        frame = np.zeros(image_np.shape, dtype=np.uint8)
        for annotation in self.annotator.annotations:
            anno_frame = self.anno_frame.copy()
            anno_frame[annotation.mask.mask != 1] = (0, 0, 0)
            frame = cv2.add(frame, anno_frame)
            # annotation.draw(frame, color=(0, 255, 0))
        updated, queued = self.annotator.step(image_np)
        if updated:
            self.anno_frame = self.next_anno_frame
        if queued:
            self.next_anno_frame = image_np.copy()

        return frame

class TrackingSuprise(Annotator):
    anno_frame = None
    def __init__(self, graph: str, label_file: str, num_classes: int, min_confidence: float = 0.5, color=None,
                 sample_delay=30, width=640, height=480):

        super().__init__(graph, label_file, num_classes, min_confidence, color, sample_delay, width=width, height=height)
        self.averager = FrameAverager(width=640, height=480, history=60)

    def apply(self, image_np):
        if self.anno_frame is None:
            self.anno_frame = image_np.copy()
        avg_frame = self.averager.apply(image_np)
        for annotation in self.annotator.annotations:
            # if annotation.label['name'] in {'book', 'cup'}:
            annotation.mask.overlay(avg_frame, self.anno_frame.copy())
            # annotation.draw(avg_frame, color=(0, 255, 0))
        updated, queued = self.annotator.step(image_np)
        if updated:
            self.anno_frame = self.next_anno_frame
        if queued:
            self.next_anno_frame = image_np.copy()

        return avg_frame