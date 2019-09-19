""" Module handling creating a stimulus out of an image.

"""

import numpy as np
import tensorflow as tf
from typing import List
from modules.grid import AbstractGrid

def value_bound(
        value: float,
        lower: float,
        upper: float
) -> float:
    """ Returns a bounded value.
    """
    if value > lower:
        if value < upper:
            return value
        return upper
    return lower


class Stimulus:
    def __init__(self, digit, image, grid, xpos=0, ypos=0):
        self.shape = image.shape

        if len(self.shape) == 2:
            self.original = image.reshape(*self.shape, 1)
            self.shape = self.original.shape
        else:
            self.original = image

        # Normalise between -1 and 1 for an RGB255 image
        if np.max(self.original) > 1:
            self.original = (self.original / 127.5) - 1

        self.padder = np.zeros((3 * self.shape[0], 3 * self.shape[1], self.shape[2])) - 1
        self.padder[self.shape[0]:2*self.shape[0], self.shape[1]:2*self.shape[1], :] = self.original

        self.digit = digit

        self.xpos = xpos
        self.ypos = ypos

        self.image = self.getImage()

        self.grid = grid
        self.sampleWidth = 4

        self.vector = self.process()

    def get_params(self, x : float, y : float):

        ymin = value_bound(int(self.shape[0] * y - self.sampleWidth // 2), 0, self.shape[0] - 1)
        ymax = value_bound(int(self.shape[0] * y + self.sampleWidth // 2), 0, self.shape[0] - 1)
        xmin = value_bound(int(self.shape[1] * x - self.sampleWidth // 2), 0, self.shape[1] - 1)
        xmax = value_bound(int(self.shape[1] * x + self.sampleWidth // 2), 0, self.shape[1] - 1)

        vals  = self.image[ymin:ymax, xmin:xmax, :]
        if 0 in vals.shape:
            return 0
        return np.mean(vals)

    def getImage(self):
        """ Based on xpos and ypos, get the image view from the padder.
        """

        xstart = self.shape[0] - int(self.xpos * self.shape[0])
        ystart = self.shape[1] - int(self.ypos * self.shape[1])

        return self.padder[ystart:ystart+self.shape[1], xstart:xstart+self.shape[0], :]

    def process(self):
        """ Converts the stimulus into a brightness vector for the
        """

        params = np.array([self.get_params(x, y) for (x, y) in self.grid.locations])
        # Normalise to between 0 and 1
        params = params - (np.min(params))
        if np.max(params) > 0:
            params /= np.max(params)
        return params

    def setPos(self, xpos: float, ypos: float):
        """Translate the image. xpos and ypos lie in the range (-1, 1)
        """
        self.xpos = xpos
        self.ypos = ypos
        self.image = self.getImage()
        self.vector = self.process()


class StimulusNet(Stimulus):

    def __init__(self, digit, image, grid, encoder):
        self.encoder = encoder
        Stimulus.__init__(self, digit, image, grid)

    def process(self):
        return self.encoder(np.array([self.digit])).numpy()[0]

def difference(rendered: np.ndarray, filled: np.ndarray) -> float:
    """
    Given the rendered and filled image, return the sum of squared pixel-wise difference.
    """

    return np.sum((rendered - filled) ** 2)

def binarise(encoding: np.ndarray, threshold: float) -> np.ndarray:
    """
    Binarises an encoding at a given threshold.
    """
    return np.where(encoding > threshold, 1.0, 0.0).astype(np.float32)

def iterate_diffs(encoding: np.ndarray, grid: AbstractGrid, filler) -> List[float]:
    """
    Given an encoding and grid, iterates through 100 thresholds between 0 and 1 at 0.01 intervals.
    """

    original = grid.render_values_tensor(encoding)
    filled = filler(tf.expand_dims(original, 0))

    diffs = []

    for threshold in np.linspace(0, 0.99, 100):
        binarised = binarise(encoding, threshold)
        rendered = grid.render_values_tensor(binarised)
        diff = difference(rendered, filled)
        diffs.append(diff)

    return diffs

def perform_best_threshold(encoder: tf.keras.Model, grid: AbstractGrid, filler, digit: int):
    """
    Given an encoder and a grid, find the best threshold for each digit for a given seed.
    """

    encoding = encoder(np.array([digit]), training=False)[0]

    diffs = iterate_diffs(encoding, grid, filler)
    lowest_diff = np.argmin(diffs)
    best_threshold = 0.01 * lowest_diff

    binarised = binarise(encoding, best_threshold)

    return binarised

class StimulusNetBinary(Stimulus):
    def __init__(self, digit, image, grid, encoder, filler):
        self.encoder = encoder
        self.pre_encoded = perform_best_threshold(encoder, grid, filler, digit)
        Stimulus.__init__(self, digit, image, grid)

    def process(self):
        return self.pre_encoded
