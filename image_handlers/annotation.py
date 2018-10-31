import cv2

import numpy as np

from annotate.video import TrackingVideoAnnotator, VideoAnnotator
from image_handlers import ImageHandler
from image_handlers.background import FrameAverager
from image_handlers.gridwall import ScalingGridWall


class Annotator(ImageHandler):

    def __init__(self, graph: str, label_file: str, num_classes: int,
                 min_confidence: float = 0.5, color=None, sample_delay=30, width=640, height=480):
        self.color = color or (0, 255, 0)
        self.annotator = VideoAnnotator(width, height, graph, label_file, num_classes, min_confidence=min_confidence,
                                        sample_delay=sample_delay)

    def handle_annotation(self, annotation, image_np):
        annotation.draw(image_np, self.color)

    def apply(self, image_np):
        self.annotator.step(image_np)
        for annotation in self.annotator.annotations:
            self.handle_annotation(annotation, image_np)
        return image_np

    def close(self):
        self.annotator.close()


class TrackingAnnotator(Annotator):

    def __init__(self, graph: str, label_file: str, num_classes: int,
                 min_confidence: float = 0.5, color=None, sample_delay=30, width=640, height=480):
        self.color = color or (0, 255, 0)
        self.annotator = TrackingVideoAnnotator(width, height, graph, label_file, num_classes,
                                                min_confidence=min_confidence,
                                                sample_delay=sample_delay)


class AnnotationGridWall(TrackingAnnotator):

    def __init__(self, *args, interesting_labels=None, overlay_time=20,
                 grid_width=3, grid_height=3, grid_padding=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.interesting_labels = interesting_labels
        self.interesting_annotations = []

        self.grid_wall = ScalingGridWall(grid_padding)
        self._max_visible = grid_height * grid_width

        self._load_annotations = True
        self._tick_update_rate = overlay_time * cv2.getTickFrequency()
        self._tick_last = cv2.getTickCount()
        self._tick_next_update = 0

    def handle_annotation(self, annotation, image_np):
        if self._load_annotations:
            if not self.interesting_labels or annotation.label['name'] in self.interesting_labels:
                self.interesting_annotations.append(annotation)

    def apply(self, image_np):
        tick = cv2.getTickCount()
        self._tick_next_update -= tick - self._tick_last
        self._tick_last = tick
        if self._tick_next_update <= 0:
            self._tick_next_update += self._tick_update_rate
            self._load_annotations = True
            self.interesting_annotations = []

        super().apply(image_np)

        if self.interesting_annotations:
            background = (image_np.copy().astype(float) * 0.3).astype(np.uint8)

            crops = []
            for a in self.interesting_annotations[:self._max_visible]:
                crops.append(cv2.cvtColor(a.crop(image_np), cv2.COLOR_BGRA2BGR))

            output = self.grid_wall.apply(background, crops)

            if self._load_annotations:
                self._load_annotations = False
        else:
            output = image_np
        return output


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

        super().__init__(graph, label_file, num_classes, min_confidence, color, sample_delay, width=width,
                         height=height)
        self.averager = FrameAverager(width=640, height=480, history=60)

    def apply(self, image_np):
        if self.anno_frame is None:
            self.anno_frame = image_np.copy()
        avg_frame = self.averager.apply(image_np)
        for annotation in self.annotator.annotations:
            annotation.mask.overlay(avg_frame, self.anno_frame.copy())
        updated, queued = self.annotator.step(image_np)
        if updated:
            self.anno_frame = self.next_anno_frame
        if queued:
            self.next_anno_frame = image_np.copy()

        return avg_frame
