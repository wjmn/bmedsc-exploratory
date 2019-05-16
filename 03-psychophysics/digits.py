#!/usr/bin/env python

################################################################################
# SETUP
################################################################################

# Imports
import numpy as np
from psychopy import visual, core, gui, data, event

from psychopy.sound.backend_pygame import SoundPygame
from psychopy.tools.filetools import fromFile, toFile
from scipy.misc import imresize
from skimage import color
from imageio import imread
from phosphenes import RegularGrid, IrregularGrid, safebound
from random import sample

################################################################################
# CONSTANTS
################################################################################

XSIZE = 480
YSIZE = 480
SCALE = 48
EXSIZE = XSIZE // SCALE
EYSIZE = YSIZE // SCALE

################################################################################
# STIMULUS
################################################################################

datadir = "data/digit-images/"
dataext = ".png"
imagesize = np.shape(
    imread(datadir + str(0) + dataext)
)  # assume all images the same size as image 0
imagescale = EXSIZE / imagesize[0]  # assuming a square image
stimuli = [
    np.fliplr(
        imresize(
            color.rgb2gray(imread(datadir + str(digit) + dataext)).transpose(),
            imagescale,
        )
    )
    for digit in range(10)
]


class Stimulus:
    def __init__(self, digit):
        self.image = stimuli[digit]
        self.vector = self.process()

    def process(self):
        """ Converts the stimulus into a brightness vector for the
        """
        flattened = self.image.flatten(order="F")
        return flattened


################################################################################
# PSYCHOPY
################################################################################

if __name__ == "__main__":

    ############################################################################
    #  SETUP
    ############################################################################

    # Expensive preparations
    # grid = RegularGrid(exsize=EXSIZE, eysize=EYSIZE)
    grid = IrregularGrid(exsize=EXSIZE, eysize=EYSIZE, randomPos=60)

    # Experiment details
    details = {"date": data.getDateStr(), "participant": ""}

    # Initial user dialog
    dialog = gui.DlgFromDict(details, title="PROTOTYPE", fixed=["date"])
    if dialog.OK:
        datafile = "./data/{}_{}_details.pickle".format(
            details["participant"], details["date"]
        )
        toFile(datafile, details)
    else:
        core.quit()

    # Clocks
    clocksession = core.Clock()
    clocktrial = core.Clock()

    # Window
    win = visual.Window([XSIZE, YSIZE])

    ntrials = 1 #30
    ncues = 20

    outfileName = f"./data/{details['participant']}_{details['date']}_session.txt"

    correctSound = SoundPygame(value='G', secs=0.1)
    incorrectSound = SoundPygame(value='Csh', secs=0.1)


    with open(outfileName, 'w+') as outfile:

        outfile.write('digit,keypress,trialtime,sessiontime\n')

        for trial in range(ntrials):

            clocktrial.reset()

            cross = visual.TextStim(win, text="+", bold=True, pos=(XSIZE/2, YSIZE/2))
            win.flip()
            event.waitKeys(clearEvents=True)

            # TODO Maybe make less predictable
            streamlists = [sample(range(10), 10) for i in range(ncues // 10)]
            stream = [i for s in streamlists for i in s]

            for cue in range(ncues):

                digit = stream.pop()
                stimulus = Stimulus(digit)
                rendered = grid.render(stimulus.vector)

                imstim = visual.ImageStim(win, image=rendered, size=(2, 2))
                imstim.draw()
                win.flip()
                keypress, *_ = event.waitKeys(timeStamped=clocktrial,
                                        clearEvents=True,
                                        keyList=[str(x) for x in range(10)])
                print(keypress)
                correct = digit == int(keypress[0])
                outfile.write(f'{str(digit)},{keypress[0]},{keypress[1]},{keypress[1]+clocksession.getTime()}\n')

                if correct:
                    correctSound.play()
                else:
                    incorrectSound.play()
