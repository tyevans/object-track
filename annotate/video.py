import time
from multiprocessing import Pipe

import cv2

from annotate.object_detector import AnnotationProcessor


class VideoAnnotator(object):

    def __init__(self, graph: str, label_file: str, num_classes: int, min_confidence: float = 0.5, sample_delay=30):
        self.min_confidence = min_confidence
        self.annotations = []

        self.sample_delay = sample_delay
        self._sample_tick = 0
        self._last_time = time.time()

        self.send_frame, self.recv_frame = Pipe()
        self.send_anno, self.recv_anno = Pipe()
        self.processor = AnnotationProcessor(graph, label_file, num_classes, self.recv_frame,
                                             self.send_anno, min_confidence=min_confidence)
        self.processor.start()

    def step(self, image_np):
        now = time.time()
        dt = now - self._last_time
        updated = False

        if self.recv_anno.poll():
            updated = True
            self.annotations = self.recv_anno.recv()

        self._sample_tick -= dt
        if self._sample_tick <= 0:
            self._sample_tick += self.sample_delay
            self.send_frame.send(image_np)
        return updated

    def close(self):
        self.send_frame.send(None)

    def __del__(self):
        self.close()


class TrackingVideoAnnotator(VideoAnnotator):

    def __init__(self, *args, tracker_func=cv2.TrackerMedianFlow_create, **kwargs):
        super(TrackingVideoAnnotator, self).__init__(*args, **kwargs)
        self.trackers = []
        self.tracker_func = tracker_func

    def step(self, image_np):
        updated = super(TrackingVideoAnnotator, self).step(image_np)
        if updated:
            self.trackers = [self.create_tracker(image_np, a) for a in self.annotations]
        else:
            for annotation, tracker in zip(self.annotations, self.trackers):
                self.update_annotation(image_np, annotation, tracker)
        return updated

    def create_tracker(self, image_np, annotation):
        height, width = image_np.shape[:2]
        y1, x1, y2, x2 = annotation.rect.translate(height, width)
        track_window = (x1, y1, x2 - x1, y2 - y1)
        tracker = self.tracker_func()
        tracker.init(image_np, track_window)
        return tracker

    def update_annotation(self, image_np, annotation, tracker):
        tracking, track_window = tracker.update(image_np)
        img_height, img_width = image_np.shape[:2]
        if tracking:
            x, y, width, height = track_window

            annotation.rect.x1 = x / img_width
            annotation.rect.x2 = (x + width) / img_width
            annotation.rect.y1 = y / img_height
            annotation.rect.y2 = (y + height) / img_height
