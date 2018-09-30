import time

import cv2
import os
from multiprocessing import Pipe
from multiprocessing import Process

import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops

from annotate.annotation import Annotation, MaskedAnnotation


class ObjectDetector:
    _tensor_dict = None

    def __init__(self, width, height, graph_pb_file, label_file, num_classes):
        self.width = width
        self.height = height
        self.graph_pb_file = graph_pb_file
        self.label_file = label_file
        self.num_classes = num_classes
        self.label_map = label_map_util.load_labelmap(self.label_file)
        self.categories = label_map_util.convert_label_map_to_categories(
            self.label_map, max_num_classes=num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph_pb_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            ops = self.detection_graph.get_operations()
            self.has_masks = any(output.name == 'detection_masks:0' for op in ops for output in op.outputs)
            self.session = tf.Session()

    @classmethod
    def load_dir(cls, graph_dir, num_classes):
        graph_pb_file = os.path.join(graph_dir, 'frozen_inference_graph.pb')
        label_file = os.path.join(graph_dir, 'labels.pbtxt')
        return cls(graph_pb_file, label_file, num_classes)

    @property
    def tensor_dict(self):
        if self._tensor_dict is None:
            graph = self.detection_graph
            self._tensor_dict = {}
            self._tensor_dict['num_detections'] = graph.get_tensor_by_name('num_detections:0')
            self._tensor_dict['detection_boxes'] = graph.get_tensor_by_name('detection_boxes:0')
            self._tensor_dict['detection_scores'] = graph.get_tensor_by_name('detection_scores:0')
            self._tensor_dict['detection_classes'] = graph.get_tensor_by_name('detection_classes:0')
            if self.has_masks:
                detection_boxes = tf.squeeze(self._tensor_dict['detection_boxes'], [0])
                detection_masks = graph.get_tensor_by_name('detection_masks:0')
                # The following processing is only for single image
                detection_masks = tf.squeeze(detection_masks, [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(self._tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, self.height, self.width)
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                self._tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
        return self._tensor_dict

    def annotate(self, image_np, min_confidence=0.5):
        # with self.detection_graph.as_default():
        # Get handles to input and output tensors
        start = time.time()
        with self.detection_graph.as_default():
            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            output_dict = self.session.run(self.tensor_dict, feed_dict={image_tensor: np.expand_dims(image_np, 0)})
        print("Inference time", time.time() - start)

        num_detections = int(output_dict['num_detections'][0])
        classes = output_dict['detection_classes'][0].astype(np.uint8)
        labels = [self.category_index.get(x, {"name": 'Unknown', "id": x}) for x in classes]
        boxes = output_dict['detection_boxes'][0].tolist()
        scores = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            masks = output_dict['detection_masks'][0]
            annotations = MaskedAnnotation.from_results(num_detections, labels, scores, boxes, masks,
                                                        min_confidence=min_confidence)
        else:
            annotations = Annotation.from_results(num_detections, labels, scores, boxes,
                                                  min_confidence=min_confidence)
        return annotations

    def close(self):
        pass


class AnnotationProcessor(Process):

    def __init__(self, width, height, graph_file: str, label_file: str, num_classes: int, recv_frame: Pipe, send_anno: Pipe,
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
        self.width = width
        self.height = height
        self.graph_file = graph_file
        self.label_file = label_file
        self.num_classes = num_classes
        self.recv_frame = recv_frame
        self.send_anno = send_anno
        self.min_confidence = min_confidence

    def run(self):
        # Create our object detector
        detector = ObjectDetector(self.width, self.height, self.graph_file, self.label_file, self.num_classes)

        while True:
            # Get the next available frame
            frame = self.recv_frame.recv()
            if frame is None:
                break
            # Annotate it
            annotations = detector.annotate(frame, min_confidence=self.min_confidence)
            # Pump it into the output queue
            self.send_anno.send(annotations)
