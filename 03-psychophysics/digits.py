#!/usr/bin/env python
"""
This script runs a digit recognition psychophysics session.
"""

# # Setup

import numpy as np
import json
import phosphenes
from datetime import datetime
from argparse import ArgumentParser
from psychopy import visual, core, gui, event
from box import Box
from psychopy.sound.backend_pygame import SoundPygame
from psychopy.tools.filetools import fromFile, toFile
from skimage import color
from imageio import imread
from random import sample
from PIL import Image

# I'm setting up a config dictionary with dot-syntax so it can be serialised 
# and saved with the session. I prefer explicitly keeping track of state.

config = Box({})

# Parsing the command line arguments, especially for testing.
parser = ArgumentParser(description='Digit recognition task.')

# Define command line arguments.
argspec = {
    'test': {
        'action': 'store_const',
        'const': True,
        'default': False,
        'dest': 'testing',
        'help': 'Test the experiment and save the data.'
    },
    'ntrials': {
        'type': int,
        'nargs': '?',
        'default': 30,
        'help': 'Number of trials for the experiment.'
    },
    'ncues': {
        'type': int,
        'nargs': '?',
        'default': 20,
        'help': 'Number of cues per trial. Should be a multiple of 10 (for now) for digit stream.'
    },
    'grid': {
        'type': str,
        'nargs': '?',
        'default': 'regular',
        'help': 'The grid type for rendering. One of regular, irregular, polarRegular or polarRegularUnique.'
    },
    'no-numpad': {
        'action': 'store_const',
        'const': True,
        'default': False,
        'dest': 'noNumpad',
        'help': 'Flags that normal number keys instead of numpad should be used.'
    }
}

# Add arguments to the parser.
[parser.add_argument(f'--{k}', **v) for k, v in argspec.items()]

# Parse the arguments and save into config.
args = parser.parse_args()
config.TESTING   = args.testing
config.NTRIALS   = args.ntrials
config.NCUES     = args.ncues
config.GRID_TYPE = args.grid
config.NO_NUMPAD = args.noNumpad


# First, we define the constants for the window size of the experiment.
# `XSIZE` and `YSIZE` refer to the size of the window on the screen.
# `EXSIZE` and `EYSIZE` refer to the size of the image data (i.e. how many 
# electrodes there are). 
# `SCALE` links the two. 

config.XSIZE  = 480
config.YSIZE  = 480
config.SCALE  = 48
config.EXSIZE = config.XSIZE // config.SCALE
config.EYSIZE = config.YSIZE // config.SCALE

# Next, we load the stimulus. Opening the image files can be expensive
# so we're doing at this at the very start and loading them into a 
# variable. 

# `IMAGE_TEMPLATE` is a string of the filepath of the stimulus digit images.
config.IMAGE_TEMPLATE = './data/digit-images/{}.png'

# `IMAGE_SIZE` is an (int, int) tuple of the image size of the first image.
# We assume that each image is of the same size as the image labelled "0"
config.IMAGE_SIZE = np.shape(imread(config.IMAGE_TEMPLATE.format(0)))  

# `IMAGE_SCALE` is an int describing the ratio of electrode size to image size.
# It assumes that EXSIZE == EYSIZE and the input images are square.
# This may need changing later. 
config.IMAGE_SCALE = config.EXSIZE / config.IMAGE_SIZE[0]  

# `IMAGES` holds the original digit images.
config.IMAGES = [np.flipud(color.rgb2gray(imread(config.IMAGE_TEMPLATE.format(digit)))) for digit in range(10)]

# `STIMULI` contains a list of numpy arrays.
# Each element in the list holds the image data (in greyscale at the moment) 
# for the digit equal to its index.
config.STIMULI = [
    np.array(Image.fromarray(image).resize((config.EXSIZE, config.EYSIZE)))
    for image in config.IMAGES
]

# We initiate a grid of electrodes.
grids = {
    'regular':   phosphenes.RegularGrid(exsize=config.EXSIZE, eysize=config.EYSIZE),
    'irregular': phosphenes.IrregularGrid(exsize=config.EXSIZE, eysize=config.EYSIZE, randomPos=0.5),
    'polarRegular': phosphenes.PolarRegularGrid(nrho=config.EXSIZE, ntheta=config.EYSIZE),
    'polarRegularUnique': phosphenes.PolarRegularUniqueGrid(nrho=config.EXSIZE, ntheta=config.EYSIZE),
}

config.GRID = grids[config.GRID_TYPE]

# Here, we define the processing method used to convert the stimulus to 
# a brightness vector for the electrodes. 
class Stimulus:
    def __init__(self, image, grid):
        self.image = image
        self.shape = self.image.shape
        self.grid = grid
        self.vector = self.process()

    def process(self):
        """ Converts the stimulus into a brightness vector for the
        """
        params = [self.image[min(self.shape[0] - 1, int(self.shape[0] * e.y)),
                             min(self.shape[1] - 1, int(self.shape[1] * e.x))] 
                             for e in self.grid.grid]
        #flattened = self.image.flatten(order="C")
        return params

# Templates for data paths.
config.DATETIME_FORMAT       = '%Y-%m-%d_%H-%M-%S'
config.DIGIT_SOUND_TEMPLATE  = './data/digit-voice/{}-alt.wav'

if config.TESTING:
    config.CONFIG_FILE_TEMPLATE  = './data/sessions/tests/{}_{}_config.json'
    config.SESSION_FILE_TEMPLATE = './data/sessions/tests/{}_{}_session.csv'
else:
    config.CONFIG_FILE_TEMPLATE  = './data/sessions/participants/{}_{}_config.json'
    config.SESSION_FILE_TEMPLATE = './data/sessions/participants/{}_{}_session.csv'

# Parameters for sound.
config.CORRECT_NOTE   = 'G'
config.INCORRECT_NOTE = 'Csh'
config.NOTE_DURATION  = 0.1
config.NOTE_VOLUME    = 0.5

# Session data.
config.SESSION_VARS = ['trial', 'cue', 'digit', 'keypress', 'cuetime', 'trialtime', 'sessiontime']

# Output templates based on session data.
config.SESSION_HEADER       = ','.join(config.SESSION_VARS) + '\n'
config.SESSION_ROW_TEMPLATE = ','.join(['{' + word + '}' for word in config.SESSION_VARS]) + '\n'

# Text.
config.PROMPT_TEXT = "{}% complete.\n\nPress any key when ready."
config.END_TEXT    = "Thank you. \n\nPress any key to exit."

# If testing, the blank image.
if config.TESTING:
    config.BLANK_FILE = config.IMAGE_TEMPLATE.format('blank')
    config.BLANK_IMAGE = np.flipud(color.rgb2gray(imread(config.BLANK_FILE)))

# Keypress during a trial.
if config.NO_NUMPAD:
    config.KEY_LIST=[str(x) for x in range(10)]
else:
    config.KEY_LIST = ["num_" + str(x) for x in range(10)]

# When saving the config, excluding some variables due to size.
config.EXCLUDED = ['STIMULI', 'GRID', 'IMAGES', 'BLANK_IMAGE']

# Here, we make our main experiment, only if called from the command line.
if __name__ == "__main__":
    
    # We initiate the user details and present a dialog to the user to get those details.
    config.details = {"datetime": datetime.strftime(datetime.now(), config.DATETIME_FORMAT), "participant": ""}
    dialog         = gui.DlgFromDict(config.details, title="PROTOTYPE", fixed=["datetime"])
    
    # We interpret the dialog actions and initiate data files if proceeding.
    if dialog.OK:
        config.configFile  = config.CONFIG_FILE_TEMPLATE.format(config.details["participant"], config.details["datetime"])
        config.sessionFile = config.SESSION_FILE_TEMPLATE.format(config.details["participant"], config.details["datetime"])
    else:
        core.quit()

    # Clocks that keep track of the experiment.
    clockSession = core.Clock()
    clockTrial   = core.Clock()
    clockCue     = core.Clock()

    # We initiate some generic sounds for correct and incorrect.
    correctSound   = SoundPygame(value=config.CORRECT_NOTE, secs=config.NOTE_DURATION)
    incorrectSound = SoundPygame(value=config.INCORRECT_NOTE, secs=config.NOTE_DURATION)
    
    correctSound.setVolume(config.NOTE_VOLUME)
    incorrectSound.setVolume(config.NOTE_VOLUME)
    
    # And we initiate the sounds for each digit.
    digitSounds = [SoundPygame(value=config.DIGIT_SOUND_TEMPLATE.format(digit)) for digit in range(10)]
    
    # Now we save the config for this session.
    with open(config.configFile, 'w+') as configFile:
        json.dump({k:v for k, v in config.items() if k not in config.EXCLUDED}, configFile)

    # We initiate a testing window if this is a testing run.
    if config.TESTING:
        testWin = visual.Window([config.XSIZE, config.YSIZE], pos=(200,200), allowGUI=False, winType='pyglet')
        win = visual.Window([config.XSIZE, config.YSIZE], pos=(200+config.XSIZE, 200), allowGUI=False, winType='pyglet', color=-1)
    else:
        # We make a window for the experiment.
        win = visual.Window(fullscr=True, allowGUI=False, winType='pyglet', color=-1)
        
    # We now start the experiment loop.
    with open(config.sessionFile, 'w+') as outfile:

        # We first write the header of the csv file.
        outfile.write(config.SESSION_HEADER)

        # Start the trial loop.
        for trial in range(config.NTRIALS):

            # Set the trial clock to 0.
            # This clock will start counting from the wait screen, so includes that time..
            clockTrial.reset()

            # If testing, show the blank.
            if config.TESTING:
                blankStimulus = Stimulus(config.BLANK_IMAGE, config.GRID)
                rendered = config.GRID.render(blankStimulus.vector)
                imageStimulus = visual.ImageStim(testWin, image=rendered, size=(2,2))
                imageStimulus.draw(); testWin.flip()

            # Show a prompt on grey background at the beginning of the trial and wait for a keypress.
            bg     = visual.GratingStim(win, tex=None, mask=None, size=2, units='norm', color=0)
            prompt = visual.TextStim(win, text=config.PROMPT_TEXT.format(trial * 100 // config.NTRIALS))
            bg.draw(); prompt.draw(); win.flip(); event.waitKeys(clearEvents=True)

            # Create a stream of digits of length NCUES for the trial.
            # At the moment, the stream is made of randomly-ordered 10-digit blocks
            # I need to think of a better way to do this to prevent duplicate digits
            # while making it more random thatn it is now.
            streamlists = [sample(range(10), 10) for i in range(config.NCUES // 10)]
            stream      = [i for s in streamlists for i in s]

            # Star the cue loop.
            for cue in range(config.NCUES):
                
                # Set the cue clock to 0.
                clockCue.reset()

                # Get a digit from the stream, make it into a stimulus and render it on the grid. 
                digit    = stream.pop()
                stimulus = Stimulus(config.STIMULI[digit], config.GRID)
                rendered = config.GRID.render(stimulus.vector)
                
                # If this is a testing run, also draw the original image.
                if config.TESTING:
                    originalImage = visual.ImageStim(testWin, image=color.rgb2gray(config.IMAGES[digit]), size=(2,2))
                    originalImage.draw(); testWin.flip()
                    
                # Create an image stimulus out of the rendered image.
                # Then show the stimulus.
                # Ensure stimulus is square on full screen window, assuming window has greater x dim than y dim.
                imstim = visual.ImageStim(win, image=rendered, size = (2 * win.size[1] / win.size[0], 2))
                imstim.draw(); win.flip()
                
                # Wait for a keypress. 
                # We only need the first keypress, and want the key input from the numpage.
                keypressRaw, *_ = event.waitKeys(clearEvents=True, keyList=config.KEY_LIST)
                
                # Check if their input was correct. 
                # Numpad keys are prepended with 'num_', so we strip it out.
                keypress = keypressRaw.strip('num_')
                correct  = (digit == int(keypress))
                
                # Create the data line.
                row = config.SESSION_ROW_TEMPLATE.format(
                    trial=trial,
                    cue=cue,
                    digit=digit,
                    keypress=keypress, 
                    cuetime=clockCue.getTime(),
                    trialtime=clockTrial.getTime(),
                    sessiontime=clockSession.getTime(),
                )
                
                # Write the data line to the session file.
                outfile.write(row)

                # Play the feedback sound.
                correctSound.play() if correct else incorrectSound.play()

                # Play the digit sound.
                digitSounds[digit].play()
                    
        # At the end of all the trials, show an end screen and wait for key press
        # to exit.
        bg  = visual.GratingStim(win, tex=None, mask=None, size=2, units='norm', color=0)
        end = visual.TextStim(win, text=config.END_TEXT)
        bg.draw(); end.draw(); win.flip(); event.waitKeys(clearEvents=True)
