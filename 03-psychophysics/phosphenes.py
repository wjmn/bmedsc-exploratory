#!/usr/bin/env python

import numpy as np
from scipy.ndimage import gaussian_filter
import random

# from scipy.ndimages.filters import convolve

# CONSTANTS

XSIZE = 480
YSIZE = 480
PBASE = 3
SCALE = 80
EXSIZE = XSIZE // SCALE
EYSIZE = YSIZE // SCALE


def safebound(value, width, lower, upper):
    vmin = int(max(lower, value - width))
    vmax = int(min(upper, value + width))
    return vmin, vmax


class Electrode:
    def __init__(self, x, y, randomPos=0):
        # self.phosphene = phosphene
        self.x = x
        self.y = y
        self.randomPos = randomPos
        self.size = PBASE * (0.5 + (4 * np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)) ** 2)

        self.rendered = self.render()

    def render(self, xsize=YSIZE, ysize=XSIZE):
        xmin, xmax = safebound(XSIZE * self.x + (random.random() - 0.5) * self.randomPos, self.size, 0, XSIZE)
        ymin, ymax = safebound(YSIZE * self.y + (random.random() - 0.5) * self.randomPos, self.size, 0, YSIZE)

        base = np.zeros((ysize, xsize))
        base[ymin:ymax, xmin:xmax] = 1

        return gaussian_filter(base, self.size)


class RegularGrid:
    def __init__(self, exsize=EXSIZE, eysize=EYSIZE):
        self.exsize = EXSIZE
        self.eysize = EYSIZE
        self.grid = [
            Electrode(y / eysize, x / exsize)
            for x in range(exsize)
            for y in range(eysize)
        ]

    def render(self, values):
        product = [v * e.rendered for (v, e) in zip(values, self.grid)]
        summed = sum(product)
        summax = np.max(summed)
        return (summed / summax) * 2 - 1

class IrregularGrid:
    def __init__(self, randomPos=2, exsize=EXSIZE, eysize=EYSIZE, ):
        self.exsize = EXSIZE
        self.eysize = EYSIZE
        self.grid = [
            Electrode(y / eysize, x / exsize, randomPos=randomPos )
            for x in range(exsize)
            for y in range(eysize)
        ]

    def render(self, values):
        product = [v * e.rendered for (v, e) in zip(values, self.grid)]
        summed = sum(product)
        summax = np.max(summed)
        return (summed / summax) * 2 - 1
