import numpy as np

import cv2


def scale_to_fit(image_np, width, height):
    img_height, img_width = image_np.shape[:2]
    resize_ratio = min(height / img_height, width / img_width)
    resize_dims = (
        int(resize_ratio * img_width),
        int(resize_ratio * img_height)
    )
    resized = cv2.resize(image_np, resize_dims, interpolation=cv2.INTER_AREA)
    return resized


class GridWall(object):

    def __init__(self, width, height, padding):
        self.width = width
        self.height = height
        self.padding = padding

    def apply(self, background, images):
        bg_height, bg_width = background.shape[:2]
        padding = self.padding

        max_visible = self.height * self.width

        overlay_width = (bg_width - (padding * (self.width + 1))) // self.width
        overlay_height = (bg_height - (padding * (self.height + 1))) // self.height
        black_frame = np.zeros((overlay_height, overlay_width, 3), dtype=np.uint8)
        for i, img_np in enumerate(images[:max_visible]):
            x = i % self.width
            y = (i - x) // self.width

            scaled_np = scale_to_fit(img_np, overlay_width, overlay_height)
            scaled_height, scaled_width = scaled_np.shape[:2]

            x_padding = (x + 1) * padding
            y_padding = (y + 1) * padding

            slot_y_top = y_padding + (overlay_height * y)
            slot_y_bottom = slot_y_top + overlay_height

            slot_x_left = x_padding + (overlay_width * x)
            slot_x_right = slot_x_left + overlay_width

            background[slot_y_top:slot_y_bottom, slot_x_left:slot_x_right] = black_frame

            auto_margin_x = (overlay_width - scaled_width) // 2
            auto_margin_y = (overlay_height - scaled_height) // 2

            left_x = slot_x_left + auto_margin_x
            right_x = left_x + scaled_width
            top_y = slot_y_top + auto_margin_y
            bottom_y = top_y + scaled_height
            background[top_y:bottom_y, left_x:right_x] = scaled_np
        return background


class ScalingGridWall(GridWall):

    def __init__(self, padding):
        self.padding = padding

    def apply(self, background, images):
        i = 1
        num_images = len(images)
        while i**2 < num_images:
            i += 1
        self.width = i
        self.height = i

        return super().apply(background, images)
