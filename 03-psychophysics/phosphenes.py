#!/usr/bin/env python

import numpy as np
from scipy.ndimage import gaussian_filter
import random
import math

# from scipy.ndimages.filters import convolve

# CONSTANTS

XSIZE = 480
YSIZE = 480
PBASE = 3
SCALE = 80
EXSIZE = XSIZE // SCALE
EYSIZE = YSIZE // SCALE


def safebound(value: float, width: float, lower: float, upper: float):
    """ 
    Returns the bounded min and max about value with width.
    """
    vmin = int(max(lower, value - width))
    vmax = int(min(upper, value + width))
    return vmin, vmax

def bound(value:float, lower: float, upper:float):
    """
    Returns a bounded value.
    """
    if value > lower:
        if value < upper:
            return value
        else:
            return upper
    else:
        return lower

class Electrode:
    def __init__(self, x: float, y: float, randomPos: float = 0):
        """
        Produces a phosphene for a single electrode.
        
        Args:
            x: float         - position in range [0, 1]. 
            y: float         - position in range [0, 1]
            randomPos: float - a scaling factor for random positioning. 
        """
        self.randomPos = randomPos
        self.x = bound(x + (random.random() - 0.5) * self.randomPos, 0, 1)
        self.y = bound(y + (random.random() - 0.5) * self.randomPos, 0, 1)

        self.size = PBASE * (0.5 + (4 * np.sqrt((self.x - 0.5) ** 2 + (self.y - 0.5) ** 2)) ** 2)

        self.rendered = self.render()

    def render(self, xsize=YSIZE, ysize=XSIZE):
        xmin, xmax = safebound(XSIZE * self.x, self.size, 0, XSIZE)
        ymin, ymax = safebound(YSIZE * self.y, self.size, 0, YSIZE)

        base = np.zeros((ysize, xsize))
        base[ymin:ymax, xmin:xmax] = 1

        return gaussian_filter(base, self.size)


class RegularGrid:
    def __init__(self, exsize: int = EXSIZE, eysize: int = EYSIZE):
        """
        
        Args:
            exsize: int - x size of electrode grid 
            eysize: int - y size of electrode grid
        """
        self.exsize = EXSIZE
        self.eysize = EYSIZE
        self.grid = [
            Electrode(x / exsize, y / eysize)
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
            Electrode(x / exsize, y / eysize, randomPos=randomPos )
            for x in range(exsize)
            for y in range(eysize)
        ]

    def render(self, values):
        product = [v * e.rendered for (v, e) in zip(values, self.grid)]
        summed = sum(product)
        summax = np.max(summed)
        return (summed / summax) * 2 - 1

class PolarRegularGrid:
    def __init__(self, nrho, ntheta):
        self.nrho   = nrho
        self.ntheta = ntheta
        self.grid = [
            # Need to think of better way to scale.
            Electrode(((math.exp(rho**0.6) / math.exp(nrho**0.6) * math.cos(2 * math.pi * theta / ntheta)) + 1) / 2,
                      ((math.exp(rho**0.6) / math.exp(nrho**0.6) * math.sin(2 * math.pi * theta / ntheta)) + 1) / 2,)
            # Ensure the central electrodes are actually visible by adding 1 to zero.
            for rho in range(1, nrho+1)
            for theta in range(ntheta)
        ]

    def render(self, values):
        product = [v * e.rendered for (v, e) in zip(values, self.grid)]
        summed = sum(product)
        summax = np.max(summed)
        return (summed / summax) * 2 - 1
