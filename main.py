import argparse

import cv2

from handlers.annotation import ImageAnnotator
from handlers.avi import AVIOutput
from handlers.background import BackgroundSubtractor
from handlers.fps import FPSCounter


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('graph')
    parser.add_argument('label_file')
    parser.add_argument('num_classes', type=int)
    parser.add_argument('--min_confidence', type=float, default=0.5)
    parser.add_argument('--show_fps', action="store_true", default=False)
    parser.add_argument('--avi_out', default=None)
    parser.add_argument('--crop_dir', default=None)
    parser.add_argument('--subtract_bg', action="store_true", default=False)
    return parser.parse_args()


class VideoSource(object):

    def __init__(self, source, processing_pipeline=[], source_capabilities=None):
        self._source = cv2.VideoCapture(source)
        self.processing_pipeline = processing_pipeline

        if source_capabilities:
            for setting, value in source_capabilities.items():
                self._source.set(setting, value)

    def run_forever(self):
        ret, frame = self._source.read()
        for handler in self.processing_pipeline:
            frame = handler.apply_first(frame)

        while (True):
            ret, frame = self._source.read()

            for handler in self.processing_pipeline:
                frame = handler.apply(frame)

            cv2.imshow('frame', frame)
            try:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    for handler in self.processing_pipeline:
                        handler.close()
                    break
            except KeyboardInterrupt:
                for handler in self.processing_pipeline:
                    handler.close()
                raise

        self._source.release()
        cv2.destroyAllWindows()


def main(options):
    pipeline = []

    if options.subtract_bg:
        pipeline.append(BackgroundSubtractor())

    pipeline.append(
        ImageAnnotator(options.graph, options.label_file, options.num_classes,
                       options.crop_dir, options.min_confidence)
    )

    if options.show_fps:
        pipeline.append(FPSCounter())

    if options.avi_out:
        pipeline.append(AVIOutput(options.avi_out))

    source = VideoSource(
        source=0,
        processing_pipeline=pipeline,
        source_capabilities={
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480
        }
    )

    source.run_forever()


if __name__ == "__main__":
    options = get_options()
    main(options)
