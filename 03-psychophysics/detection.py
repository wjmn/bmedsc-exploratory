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


    ############################################################################
    #  SETUP
    ############################################################################

    # Expensive preparations
    grid = RegularGrid()

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
    win = visual.Window([640,480])

    # Mouse
    mouse = event.Mouse(win=win)

    ntrials = 2
    nframes = 60

    ############################################################################
    # INITIAL
    ############################################################################

    # incentre = False
    # while not incentre:
    #     pos = mouse.getPos()
    #     print(pos)
    #     if 0 < abs(pos[0]) < 0.5 and 0 < abs(pos[1]) < 0.5:
    #         break

    ############################################################################
    # START
    ############################################################################

    rate = 0.02

    tone = SoundPygame(secs=0.1)

    with open(datafile.replace("details", "data"), "w+") as outfile:
        outfile.write("trial,frame,xpos,ypos,mousexpos,mouseypos\n")

        for t in range(ntrials):

            intro = visual.TextStim(win, pos=[0,-100],text='Click the center when ready.')
            fixation = visual.ShapeStim(win, lineColor=-1, lineColorSpace='rgb', vertices="cross", fillColor=-1, size=0.2)

            intro.draw()
            fixation.draw()
            win.flip()
            event.waitKeys()

            mouse.setPos((0,0))

            xpos, ypos = 0.5, 0.5

            for i in range(nframes):

                pos = mouse.getPos()
                mypos = (pos[0] + 1) / 2
                mxpos = (pos[1] + 1) / 2
                datastring = "{},{},{},{},{},{}\n".format(t,i,xpos,ypos,mxpos,mypos)
                if abs(xpos - mxpos) > 0.1 or abs(ypos - mypos) > 0.1:
                    tone.play()
                print(datastring)
                outfile.write(datastring)

                direction = np.random.uniform(2*np.pi)
                xshift, yshift = rate*np.cos(direction), rate*np.sin(direction)
                xpos += xshift
                ypos += yshift

                stimulus = Stimulus(xpos, ypos)
                rendered = grid.render(stimulus.vector)

                imstim = visual.ImageStim(win, image=rendered, size=(2,2))
                imstim.draw()
                win.flip()
