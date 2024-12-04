import cv2
import numpy as np

# Replace 'path_to_image.jpg' with the actual image path
image = cv2.imread('./Original-Form-2.jpg')

# Keep a copy of the original image for later use
orig_image = image.copy()


def resize_image(image, max_height=800):
    # image.shape returns a tuple of image dimension, height width channel
    # we grab the first two elements of the tuple height and width
    height, width = image.shape[:2]
    if height > max_height:
        ratio = max_height / float(height)
        new_dimensions = (int(width * ratio), max_height)
        image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    return image

image = resize_image(image)

