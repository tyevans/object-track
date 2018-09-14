# import the necessary packages
import numpy as np


# Malisiewicz et al.
# Implementation stolen and tweaked from
# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression(annotations, overlapThresh):
    if not annotations:
        return []

    all_annotations = []
    annotations_by_label = {}
    for a in annotations:
        annotations_by_label.setdefault(a.label['name'], []).append(a)

    for anno_group in annotations_by_label.values():
        boxes = np.array([anno.rect.to_array() for anno in anno_group])

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        if len(pick) <= 1:
            anno = anno_group[pick[0]]
            if anno.color != (0, 255, 0):
                all_annotations.append(anno)
            else:
                anno.track_miss += 1
                if anno.track_miss <= 15:
                    all_annotations.append(anno)
        else:
            for p in pick:
                anno = anno_group[p]
                anno.track_miss = 0
                if anno not in all_annotations:
                    all_annotations.append(anno)
    return all_annotations
