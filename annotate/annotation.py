import cv2

from annotate.mask import Mask
from annotate.rect import Rect


class Annotation(object):

    def __init__(self, label, score, box):
        self.label = label
        self.rect = Rect(box[0], box[1], box[2], box[3])
        self.score = score

    def draw(self, image_np, color, draw_label=True):
        height, width = image_np.shape[:2]
        self.rect.draw(image_np, color)

        if draw_label:
            font = cv2.FONT_HERSHEY_PLAIN
            fontScale = 1
            lineType = 1
            padding = 5
            d_padding = padding * 2
            label = "{} ({:.2f})".format(self.label['name'], self.score)
            text_dims, _ = cv2.getTextSize(label, font, fontScale, lineType)
            text_width, text_height = text_dims
            x = int(self.rect.x1 * width)
            y = int(self.rect.y1 * height)
            cv2.rectangle(image_np, (x, y - text_height - d_padding), (x + text_width + d_padding, y), color, -1)
            cv2.putText(image_np, label, (x + padding, y - padding), font, fontScale, (0, 0, 0), lineType)

    @classmethod
    def from_results(cls, num_detections, labels, scores, boxes, min_confidence=0.5):
        annotations = []
        zipped = zip(labels[:num_detections], scores[:num_detections], boxes[:num_detections])
        for i, (label, score, box) in enumerate(zipped):
            if score >= min_confidence:
                annotations.append(cls(label, score, box))
        return annotations

    def crop(self, image_np):
        return self.rect.crop(image_np)


class MaskedAnnotation(Annotation):

    def __init__(self, label, score, box, mask=None):
        super(MaskedAnnotation, self).__init__(label, score, box)
        self.mask = Mask(mask if mask is not None else [])

    def crop(self, image_np):
        transparent = self.mask.as_transparency(image_np)
        return self.rect.crop(transparent)

    def draw(self, image_np, color, draw_label=True, draw_rect=True):
        self.mask.draw(image_np, color)
        if draw_rect:
            self.rect.draw(image_np, color)

    @classmethod
    def from_results(cls, num_detections, labels, scores, boxes, masks=None, min_confidence=0.5):
        masks = masks if masks is not None else []
        annotations = []
        zipped = zip(labels[:num_detections], scores[:num_detections], boxes[:num_detections], masks[:num_detections])
        for i, (label, score, box, mask) in enumerate(zipped):
            if score >= min_confidence:
                annotations.append(cls(label, score, box, mask))
        return annotations

    def __str__(self):
        return "{} :: {}".format(self.label, self.score)