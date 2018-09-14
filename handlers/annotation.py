from __future__ import division

import queue
import uuid
from multiprocessing import Process, Queue

import cv2

from detect.annotation import Rect
from detect.object_detector import ObjectDetector
from detect.util import non_max_suppression
from handlers import ImageHandler


class AnnotationProcessor(Process):

    def __init__(self, graph_file: str, label_file: str, num_classes: int, frame_queue: Queue, annotation_queue: Queue,
                 min_confidence: float = 0.5):
        """ Dedicated image annotating process.  Sits off to the side

        Pump `None` into the frame queue to exit the process

        :param graph_file: (string) Path to the object detection model's frozen_inference_graph.pb
        :param label_file: (string) Path to the object detection model's labels.pbtxt
        :param num_classes: (int) Number of classes the model is trained on (this is 90 for a coco trained model)
        :param frame_queue: (multiprocessing.Queue) queue where raw images will be provided
        :param annotation_queue: (multiprocessing.Queue) queue where annotations will be returned
        """
        super(AnnotationProcessor, self).__init__()
        self.graph_file = graph_file
        self.label_file = label_file
        self.num_classes = num_classes
        self.frame_queue = frame_queue
        self.annotation_queue = annotation_queue
        self.min_confidence = min_confidence

    def run(self):
        # Create our object detector
        detector = ObjectDetector(self.graph_file, self.label_file, self.num_classes)

        while True:
            # Get the next available frame
            frame = self.frame_queue.get()
            if frame is None:
                break
            # Annotate it
            annotations = detector.annotate(frame, min_confidence=self.min_confidence)
            # Pump it into the output queue
            self.annotation_queue.put(annotations)


class TrackingAnnotation(object):

    def __init__(self, image_np, annotation):
        """ `Annotation` wrapper that applies meanshift to keep an annotation centered on its target.

        :param image_np: initial image
        :param annotation: `Annotation` instance
        """
        self.uuid = str(uuid.uuid4())
        self.annotation = annotation
        self.height, self.width = image_np.shape[:2]
        self.color = (0, 0, 255)
        self.tracker = cv2.TrackerMOSSE_create()
        self.init_tracker(image_np)

    def init_tracker(self, image_np):
        y1, x1, y2, x2 = self.annotation.rect.translate(self.height, self.width)
        self.track_window = (x1, y1, x2 - x1, y2 - y1)

        self.tracker.clear()
        self.tracker.init(image_np, self.track_window)

        r = self.annotation.rect
        self.start_rect = Rect(r.y1, r.x1, r.y2, r.x2)
        self.tracking = True
        self.track_miss = 0

    def step(self, image_np):
        """ Updates the wrapped annotation based on the data present in `image_np`

        :param image_np: the current image frame
        """
        if not self.tracking:
            return

        self.color = (0, 255, 0)

        self.tracking, self.track_window = self.tracker.update(image_np)
        if self.tracking:
            x, y, width, height = self.track_window

            self.annotation.rect.x1 = x / self.width
            self.annotation.rect.x2 = (x + width) / self.width

            self.annotation.rect.y1 = y / self.height
            self.annotation.rect.y2 = (y + height) / self.height

    def __getattr__(self, item):
        """ Present so the underlying annotation object's content is accessible.
        """
        return getattr(self.annotation, item)


class ImageAnnotator(ImageHandler):

    def __init__(self, graph, label_file, num_class, crop_dir=None, min_confidence=0.5):
        """ Annotates images using an `AnnotationProcessor` subprocess

        :param graph_file: (string) Path to the object detection model's frozen_inference_graph.pb
        :param label_file: (string) Path to the object detection model's labels.pbtxt
        :param num_classes: (int) Number of classes the model is trained on (this is 90 for a coco trained model)
        :param crop_dir: (string) Path to directory where annotation crops should be stored
        :param min_confidence: (float) Minimum confidence score for annotations
        """
        self.crop_dir = crop_dir
        self.min_confidence = min_confidence
        self.frame_queue = Queue()
        self.annotation_queue = Queue()
        self.processor = AnnotationProcessor(graph, label_file, num_class, self.frame_queue,
                                             self.annotation_queue, min_confidence=min_confidence)
        self.processor.start()
        self.prev_annotations = []
        self.annotations = []

    def apply_first(self, image_np):
        self.anno_frame = image_np.copy()
        self.frame_queue.put(image_np)
        return image_np

    def apply(self, image_np):
        try:
            annotations = [TrackingAnnotation(self.anno_frame, a) for a in self.annotation_queue.get_nowait()]
            self.annotations = [a for a in self.annotations if a.tracking]
            self.annotations = [a for a in non_max_suppression(annotations + self.annotations, 0.90)]
        except queue.Empty:
            for annotation in self.annotations:
                annotation.step(image_np)
        else:
            self.anno_frame = image_np.copy()
            self.frame_queue.put(image_np)
        self.draw_annotations(image_np)

        return image_np

    def close(self):
        self.frame_queue.put(None)

    def draw_annotations(self, image_np):
        for annotation in self.annotations:
            if annotation.score >= self.min_confidence:
                annotation.draw(image_np, annotation.color)
