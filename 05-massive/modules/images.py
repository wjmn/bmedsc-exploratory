import tensorflow as tf
import random
import numpy as np
import cv2
import modules.grid


class Images:

    images: np.ndarray
    labels: np.ndarray
    length: int

    def preprocess_for_grid(self, grid: modules.grid.AbstractGrid) -> None:
        if grid.half:
            render_y, render_x = grid.render_shape
            render_x_half = render_x // 2
            render_x_rem = render_x - render_x_half

            processed = np.array([
                np.pad(
                    cv2.resize(image, (render_x_rem, render_y)),
                    ((0, 0), (render_x_half, 0)),  # Pad on the left hand-side
                    'constant',
                    constant_values=0
                )
                for image in self.images
            ])
        else:
            processed = np.array([cv2.resize(image, grid.render_shape) for image in self.images])

        if processed.max() != 0:                      # Normalise between -1 and 1
            processed = (processed - processed.min())
            processed = processed / processed.max()
            processed = (processed * 2) - 1

        self.images = processed


class Mnist(Images):

    def __init__(self) -> None:

        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

        concat = list(zip(tf.concat((train_images, test_images), 0), tf.concat((train_labels, test_labels), 0)))
        random.shuffle(concat)

        self.images, self.labels = zip(*concat)
        self.images = tf.stack(self.images, axis=0).numpy()
        self.labels = tf.stack(self.labels, axis=0).numpy()
        assert len(self.images) == len(self.labels)
        self.length = len(self.images)
