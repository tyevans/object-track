from annotate.video import TrackingVideoAnnotator, VideoAnnotator
from image_handlers import ImageHandler
from image_handlers.background import FrameAverager


class Annotator(ImageHandler):

    def __init__(self, graph: str, label_file: str, num_classes: int,
                 min_confidence: float = 0.5, color=None, sample_delay=30):
        self.color = color or (0, 255, 0)
        self.annotator = VideoAnnotator(graph, label_file, num_classes, min_confidence=min_confidence,
                                        sample_delay=sample_delay)

    def apply(self, image_np):
        self.annotator.step(image_np)
        for annotation in self.annotator.annotations:
            print(annotation)
            annotation.draw(image_np, self.color)
        return image_np

    def close(self):
        self.annotator.close()


class TrackingAnnotator(Annotator):

    def __init__(self, graph: str, label_file: str, num_classes: int,
                 min_confidence: float = 0.5, color=None, sample_delay=30):
        self.color = color or (0, 255, 0)
        self.annotator = TrackingVideoAnnotator(graph, label_file, num_classes, min_confidence=min_confidence,
                                                sample_delay=sample_delay)

class TrackingSuprise(Annotator):
    anno_frame = None
    def __init__(self, graph: str, label_file: str, num_classes: int, min_confidence: float = 0.5, color=None,
                 sample_delay=30):
        super().__init__(graph, label_file, num_classes, min_confidence, color, sample_delay)
        self.averager = FrameAverager(history=20)

    def apply(self, image_np):
        if self.anno_frame is None:
            self.anno_frame = image_np.copy()
        avg_frame = self.averager.apply(image_np)
        updated, queued = self.annotator.step(image_np)
        for annotation in self.annotator.annotations:
            if annotation.label['name'] not in {'book', 'cup', 'dog', 'cat'}:
                continue
            annotation.mask.overlay(avg_frame, self.anno_frame.copy())
            # annotation.draw(avg_frame, color=(0, 255, 0))
        if updated:
            self.anno_frame = image_np.copy()
        return avg_frame