#!/usr/bin/env python

################################################################################
# SETUP
################################################################################

# Imports
import numpy as np
from psychopy import visual, core, gui, data, event
#from psychopy.sound.backend_pygame import SoundPygame
from psychopy.tools.filetools import fromFile, toFile
from scipy.misc import imresize
from skimage import color
from imageio import imread
from phosphenes import RegularGrid, safebound
from random import randint

################################################################################
# CONSTANTS
################################################################################

XSIZE = 480
YSIZE = 480
SCALE = 30
EXSIZE = XSIZE // SCALE
EYSIZE = YSIZE // SCALE

################################################################################
# STIMULUS
################################################################################

datadir = 'data/digit-images/'
dataext = '.png'
imagesize = np.shape(imread(datadir+str(0)+dataext)) # assume all images the same size as image 0
imagescale = EXSIZE / imagesize[0] # assuming a square image
stimuli = [np.fliplr(imresize(color.rgb2gray(imread(datadir + str(digit) + dataext)).transpose(), imagescale)) for digit in range(10)]

class Stimulus:
    def __init__(self, digit):
        self.image = stimuli[digit]
        self.vector = self.process()

    def process(self):
        """ Converts the stimulus into a brightness vector for the
        """
        flattened = self.image.flatten(order='F')
        return flattened


################################################################################
# PSYCHOPY
################################################################################

if __name__ == "__main__":

    ############################################################################
    #  SETUP
    ############################################################################

    # Expensive preparations
    grid = RegularGrid(exsize=EXSIZE, eysize=EYSIZE)

    # Experiment details
    details = {
        "date": data.getDateStr(),
        "participant": "",
    }

    # Initial user dialog
    dialog = gui.DlgFromDict(details, title='PROTOTYPE', fixed=['date'])
    if dialog.OK:
        datafile =  "./data/{}_{}_details.pickle".format(details["participant"], details["date"])
        toFile(datafile, details)
    else:
        core.quit()

    # Clocks
    clockglobal = core.Clock()

    clocktrial = core.Clock()

    # Window
    win = visual.Window([XSIZE,YSIZE])

    ntrials = 30
    ncues   = 20

    for trial in range(ntrials):

        for cue in range(ncues):

            digit = randint(0, 9)
            stimulus = Stimulus(digit)
            rendered = grid.render(stimulus.vector)

            imstim = visual.ImageStim(win, image=rendered, size=(2,2))
            imstim.draw()
            win.flip()
            event.waitKeys()
