import cv2
import numpy as np


class Rect(object):

    def __init__(self, y1, x1, y2, x2):
        self.x1 = x1
        self.y1 = y1

        self.x2 = x2
        self.y2 = y2

    @property
    def centroid(self):
        return (self.x2 - self.x1, self.y2 - self.y1)

    def translate(self, height, width):
        x1 = int(self.x1 * width)
        y1 = int(self.y1 * height)
        x2 = int(self.x2 * width)
        y2 = int(self.y2 * height)
        return y1, x1, y2, x2

    def crop(self, image_np):
        y1, x1, y2, x2 = self.translate(*image_np.shape[:2])
        return image_np[y1:y2, x1:x2]

    def draw(self, image_np, color, line_width=2):
        y1, x1, y2, x2 = self.translate(*image_np.shape[:2])
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, line_width)

    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def overlap(self, other):
        area = self.area()
        dx = min(self.x2, other.x2) - max(self.x1, other.x1)
        dy = min(self.y2, other.y2) - max(self.y1, other.y1)
        if (dx >= 0) and (dy >= 0):
            return dx * dy

    def union(self, other):
        overlap = self.overlap(other) or 0
        return self.area() + other.area() - overlap

    def iou(self, other):
        union = self.union(other)
        return (self.overlap(other) or 0) / union

    def to_array(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def __str__(self):
        return "{}, {}, {}, {}".format(self.x1, self.y1, self.x2, self.y2)
