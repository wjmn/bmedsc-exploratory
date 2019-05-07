#!/usr/bin/env python

################################################################################
# SETUP
################################################################################

# Imports
import numpy as np
from psychopy import visual, core
from scipy.misc import imresize
from phosphenes import RegularGrid, safebound

################################################################################
# CONSTANTS
################################################################################

XSIZE = 640
YSIZE = 480
SCALE = 80
EXSIZE = XSIZE // SCALE
EYSIZE = YSIZE // SCALE
SSIZE = 50

################################################################################
# STIMULUS
################################################################################

class Stimulus:
    def __init__(self, xpos, ypos):
        self.xpos = xpos
        self.ypos = ypos

        self.image = self.make()
        self.vector = self.process()


    def make(self):
        base = np.zeros((YSIZE, XSIZE))
        xmin, xmax = safebound(self.xpos*XSIZE, SSIZE, 0, XSIZE)
        ymin, ymax = safebound(self.ypos*YSIZE, SSIZE, 0, YSIZE)
        base[ymin:ymax, xmin:xmax] = 1
        return base

    def process(self):
        downsampled = imresize(self.image, 1/SCALE)
        flattened = downsampled.flatten(order='F')
        return flattened


################################################################################
# PSYCHOPY
################################################################################

if __name__ == "__main__":

    # Window
    window = visual.Window()

    # Stimulus
    grid = RegularGrid()

    for i in range(100):
        xpos, ypos = np.random.random(2)

        stimulus = Stimulus(xpos, ypos)
        rendered = grid.render(stimulus.vector)

        imstim = visual.ImageStim(window, image=rendered)
        imstim.draw()
        window.flip()
