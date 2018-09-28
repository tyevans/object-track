import argparse

import cv2

from image_handlers import VideoDisplay
from image_handlers.annotation import AnnotatingImageHandler, SimpleAnnotatingImageHandler
from image_handlers.avi import AVIOutput
from image_handlers.fps import FPSCounter


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('graph')
    parser.add_argument('label_file')
    parser.add_argument('num_classes', type=int)
    parser.add_argument('--min_confidence', type=float, default=0.5)
    parser.add_argument('--show_fps', action="store_true", default=False)
    parser.add_argument('--avi_out', default=None)
    parser.add_argument('--crop_dir', default=None)
    parser.add_argument('--i_width', default=640, type=int)
    parser.add_argument('--i_height', default=480, type=int)
    return parser.parse_args()


class VideoSource(object):

    def __init__(self, source, processing_pipeline=[], source_capabilities=None):
        self._source = cv2.VideoCapture(source)
        self.processing_pipeline = processing_pipeline

        if source_capabilities:
            for setting, value in source_capabilities.items():
                self._source.set(setting, value)

    def run_forever(self):
        while (True):
            ret, frame = self._source.read()

            for handler in self.processing_pipeline:
                frame = handler.apply(frame)
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


if __name__ == "__main__":
    options = get_options()

    pipeline = [
        AnnotatingImageHandler(options.graph, options.label_file, options.num_classes,
                               min_confidence=options.min_confidence, sample_delay=30)
    ]

    if options.show_fps:
        pipeline.append(FPSCounter())

    if options.avi_out:
        pipeline.append(AVIOutput(options.avi_out, width=options.i_width, height=options.i_height, frame_rate=15.0))

    pipeline.append(VideoDisplay("Frame"))

    source = VideoSource(
        source=0,
        processing_pipeline=pipeline,
        source_capabilities={
            cv2.CAP_PROP_FRAME_WIDTH: options.i_width,
            cv2.CAP_PROP_FRAME_HEIGHT: options.i_height,
            cv2.CAP_PROP_AUTO_EXPOSURE: 1,
            cv2.CAP_PROP_AUTOFOCUS: 1,
            # cv2.CAP_PROP_BRIGHTNESS: 0.5,
            # cv2.CAP_PROP_CONTRAST: 0.5,
        }
    )
    source.run_forever()
