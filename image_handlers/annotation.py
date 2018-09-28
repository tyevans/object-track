from annotate.video import TrackingVideoAnnotator, VideoAnnotator
from image_handlers import ImageHandler


class SimpleAnnotatingImageHandler(ImageHandler):

    def __init__(self, graph: str, label_file: str, num_classes: int,
                 min_confidence: float = 0.5, color=None, sample_delay=30):
        self.color = color or (0, 255, 0)
        self.annotator = VideoAnnotator(graph, label_file, num_classes, min_confidence=min_confidence,
                                        sample_delay=sample_delay)

    def apply(self, image_np):
        self.annotator.step(image_np)
        for annotation in self.annotator.annotations:
            annotation.draw(image_np, self.color)
        return image_np

    def close(self):
        self.annotator.close()


class AnnotatingImageHandler(SimpleAnnotatingImageHandler):

    def __init__(self, graph: str, label_file: str, num_classes: int,
                 min_confidence: float = 0.5, color=None, sample_delay=30):
        self.color = color or (0, 255, 0)
        self.annotator = TrackingVideoAnnotator(graph, label_file, num_classes, min_confidence=min_confidence,
                                                sample_delay=sample_delay)
