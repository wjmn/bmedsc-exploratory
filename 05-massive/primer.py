#!/usr/bin/env python
"""
This script runs a digit recognition psychophysics priming session ( no feedback , multiple grids ).
"""

# # Setup

import numpy as np
import json
import cv2
import pickle
import modules.grid
import modules.stimulus
import tensorflow as tf
from modules.config import Config
from modules.stimulus import Stimulus, StimulusNet, StimulusNetBinary
from datetime import datetime
from argparse import ArgumentParser
from psychopy import visual, core, gui, event
from box import Box
from psychopy.sound.backend_pygame import SoundPygame
from skimage import color
from imageio import imread
from random import randint, random
from PIL import Image


# I'm setting up a config dictionary with dot-syntax so it can be serialised
# and saved with the session. I prefer explicitly keeping track of state.

config = Box({})

# Parsing the command line arguments, especially for testing.
parser = ArgumentParser(description='Digit recognition task.')

# Define command line arguments.
argspec = {
    'testing': {
        'action': 'store_const',
        'const': True,
        'default': False,
        'dest': 'testing',
        'help': 'Test the experiment and save the data.'
    },
    'ntrials': {
        'type': int,
        'nargs': '?',
        'default': 20,
        'help': 'Number of trials for the experiment.'
    },
    'ncues': {
        'type': int,
        'nargs': '?',
        'default': 25,
        'help': 'Number of cues per trial. Should be a multiple of 10 (for now) for digit stream.'
    },
    'checkpoint-num': {
        'type': int,
        'nargs': '?',
        'help': 'Encoder checkpoint to use.'
    },
    'config-id': {
        'type': str,
        'nargs': '?',
        'default': None,
        'help': 'The ID of the config to run the psychophysics experiment over.'
    },
    'binarised': {
        'action': 'store_const',
        'const': True,
        'default': False,
        'dest': 'binarised',
        'help': 'Flags whether to use binarised encodings for test condition.'
    },
    'no-numpad': {
        'action': 'store_const',
        'const': True,
        'default': False,
        'dest': 'noNumpad',
        'help': 'Flags that normal number keys instead of numpad should be used.'
    },
    'with-scanning': {
        'action': 'store_const',
        'const': True,
        'default': False,
        'dest': 'withScanning',
        'help': 'Flags that scanning with the mouse should be enabled.'
    }
}

# Add arguments to the parser.
[parser.add_argument(f'--{k}', **v) for k, v in argspec.items()]

# Parse the arguments and save into config.
args = parser.parse_args()
config.TESTING = args.testing
config.NTRIALS = args.ntrials
config.NCUES = args.ncues
config.CONFIG_ID = args.config_id
config.CHECKPOINT_NUM = args.checkpoint_num
config.BINARISED = args.binarised
config.NO_NUMPAD = args.noNumpad
config.WITH_SCANNING = args.withScanning

# Load the training configuration file.
TRAINING_CONFIG = Config.load('.', config.CONFIG_ID)

# First, we define the constants for the window size of the experiment.
# `XSIZE` and `YSIZE` refer to the size of the window on the screen.
config.YSIZE, config.XSIZE = 100, 100

# Next, we load the stimulus. Opening the image files can be expensive
# so we're doing at this at the very start and loading them into a
# variable.

# `IMAGE_TEMPLATE` is a string of the filepath of the stimulus digit images.
config.IMAGE_TEMPLATE = './images/{}.png'

# `IMAGE_SIZE` is an (int, int) tuple of the image size of the first image.
# We assume that each image is of the same size as the image labelled "0"
config.IMAGE_SIZE = np.shape(imread(config.IMAGE_TEMPLATE.format(0)))

# `IMAGES` holds the original digit images.
config.IMAGES = [cv2.cvtColor(cv2.resize(imread(config.IMAGE_TEMPLATE.format(digit)),
                                         dsize=(TRAINING_CONFIG.grid.render_shape)),
                              cv2.COLOR_RGBA2RGB)
                            for digit in range(10)]


# Initialise the two processors
config.PROCESSOR_CONTROL = Stimulus

LEARNING_RATE_ENCODER = TRAINING_CONFIG.learning_rate_encoder
LEARNING_RATE_DECODER = TRAINING_CONFIG.learning_rate_decoder
ENCODER_OPTIMISER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_ENCODER)
DECODER_OPTIMISER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_DECODER)

CHECKPOINT = tf.train.Checkpoint(
    encoder=TRAINING_CONFIG.encoder,
    decoder=TRAINING_CONFIG.decoder,
    encoder_optimiser=ENCODER_OPTIMISER,
    decoder_optimiser=DECODER_OPTIMISER,
)

config.CHECKPOINT_TEMPLATE = f'./output/{TRAINING_CONFIG.config_id}/checkpoints/checkpoint_-{config.CHECKPOINT_NUM}'
CHECKPOINT.restore(config.CHECKPOINT_TEMPLATE)

encoder = CHECKPOINT.encoder

if config.BINARISED:
    config.PROCESSOR_TEST = lambda digit, image, grid: StimulusNetBinary(digit, image, grid, encoder, TRAINING_CONFIG.filler)
else:
    config.PROCESSOR_TEST = lambda digit, image, grid: StimulusNet(digit, image, grid, encoder)


# Templates for data paths.
config.DATETIME_FORMAT       = '%Y-%m-%d_%H-%M-%S'
config.DIGIT_SOUND_TEMPLATE  = './sounds/{}.wav'

base_dir = './experiments/'

if config.TESTING:
    config.CONFIG_FILE_TEMPLATE  = base_dir + 'tests/{}_{}_config.json'
    config.SESSION_FILE_TEMPLATE = base_dir + 'tests/{}_{}_session.csv'
    config.MOUSE_FILE_TEMPLATE   = base_dir + 'tests/{}_{}_mouse.csv'
else:
    config.CONFIG_FILE_TEMPLATE  = base_dir + 'participants/{}_{}_config.json'
    config.SESSION_FILE_TEMPLATE = base_dir + 'participants/{}_{}_session.csv'
    config.MOUSE_FILE_TEMPLATE   = base_dir + 'participants/{}_{}_mouse.csv'

# Parameters for sound.
config.CORRECT_NOTE   = 'G'
config.INCORRECT_NOTE = 'Csh'
config.NOTE_DURATION  = 0.1
config.NOTE_VOLUME    = 0.5

# Session data.
config.SESSION_VARS = ['trial', 'cue', 'digit', 'processor', 'keypress', 'cuetime', 'trialtime', 'sessiontime']
config.MOUSE_VARS   = ['trial', 'cue', 'digit', 'processor', 'xmouse', 'ymouse', 'cuetime', 'trialtime', 'sessiontime']

# Output templates based on session data.
config.SESSION_HEADER       = ','.join(config.SESSION_VARS) + '\n'
config.SESSION_ROW_TEMPLATE = ','.join(['{' + word + '}' for word in config.SESSION_VARS]) + '\n'
config.MOUSE_HEADER         = ','.join(config.MOUSE_VARS) + '\n'
config.MOUSE_ROW_TEMPLATE   = ','.join(['{' + word + '}' for word in config.MOUSE_VARS]) + '\n'

# Mouse recording interval in seconds.
config.MOUSE_RECORD_INTERVAL = 0.2

# Text.
config.PROMPT_TEXT = "{}% complete.\n\nPress any key when ready."
config.END_TEXT    = "Thank you. \n\nPress any key to exit."

# If testing, the blank image.
if config.TESTING:
    #config.BLANK_FILE = config.IMAGE_TEMPLATE.format('blank')
    #config.BLANK_IMAGE = cv2.resize(np.flipud(imread(config.BLANK_FILE)), dsize=(config.INPUT_XSIZE, config.INPUT_YSIZE))
    config.TEST_WINDOW_XSIZE = 480
    config.TEST_WINDOW_YSIZE = 480

# Keypress during a trial.
if config.NO_NUMPAD:
    config.KEY_LIST=[str(x) for x in range(10)]
else:
    config.KEY_LIST = ["num_" + str(x) for x in range(10)]

# When saving the config, excluding some variables due to size.
config.EXCLUDED = ['STIMULI', 'GRID', 'IMAGES', 'BLANK_IMAGE', 'PROCESSOR_TEST', 'PROCESSOR_CONTROL']


# Here, we make our main experiment, only if called from the command line.
if __name__ == "__main__":

    # We initiate the user details and present a dialog to the user to get those details.
    config.details = {"datetime": datetime.strftime(datetime.now(), config.DATETIME_FORMAT), "participant": ""}
    dialog         = gui.DlgFromDict(config.details, title="PROTOTYPE", fixed=["datetime"])

    # We interpret the dialog actions and initiate data files if proceeding.
    if dialog.OK:
        config.configFile  = config.CONFIG_FILE_TEMPLATE.format(config.details["participant"], config.details["datetime"])
        config.sessionFile = config.SESSION_FILE_TEMPLATE.format(config.details["participant"], config.details["datetime"])
        config.mouseFile = config.MOUSE_FILE_TEMPLATE.format(config.details["participant"], config.details["datetime"])
    else:
        core.quit()

    # Clocks that keep track of the experiment.
    clockSession = core.Clock()
    clockTrial   = core.Clock()
    clockCue     = core.Clock()
    mouseRecord  = core.Clock()

    # We initiate some generic sounds for correct and incorrect.
    correctSound   = SoundPygame(value=config.CORRECT_NOTE, secs=config.NOTE_DURATION)
    incorrectSound = SoundPygame(value=config.INCORRECT_NOTE, secs=config.NOTE_DURATION)

    correctSound.setVolume(config.NOTE_VOLUME)
    incorrectSound.setVolume(config.NOTE_VOLUME)

    # And we initiate the sounds for each digit.
    digitSounds = [SoundPygame(value=config.DIGIT_SOUND_TEMPLATE.format(digit)) for digit in range(10)]

    # Now we save the config for this session.
    with open(config.configFile, 'w+') as configFile:
        json.dump({k:v for k, v in config.items() if k not in config.EXCLUDED}, configFile, sort_keys=True, indent=4)

    # We initiate a testing window if this is a testing run.
    if config.TESTING:
        testWin = visual.Window([config.TEST_WINDOW_XSIZE, config.TEST_WINDOW_YSIZE],
                                pos=(200,200), allowGUI=False, winType='pyglet')
        win = visual.Window([config.TEST_WINDOW_XSIZE, config.TEST_WINDOW_YSIZE],
                            pos=(200+config.TEST_WINDOW_XSIZE, 200), allowGUI=False, winType='pyglet', color=-1)
    else:
        # We make a window for the experiment.
        win = visual.Window(fullscr=True, allowGUI=False, winType='pyglet', color=-1)

    # Start the mouse event
    mouse = event.Mouse(visible=False, win=win)

    # We now start the experiment loop.
    with open(config.sessionFile, 'w+') as outfile: #, open(config.mouseFile, 'w+') as mousefile:

        # We first write the header of the csv file.
        outfile.write(config.SESSION_HEADER)
        # mousefile.write(config.MOUSE_HEADER)

        # Start the trial loop.
        for trial in range(config.NTRIALS):

            # Set the trial clock to 0.
            # This clock will start counting from the wait screen, so includes that time..
            clockTrial.reset()

            # If testing, show the blank.
            # if config.TESTING:
            #     blankStimulus = config.PROCESSOR(config.BLANK_IMAGE, config.GRID)
            #     rendered = config.GRID.render(blankStimulus.vector)
            #     imageStimulus = visual.ImageStim(testWin, image=rendered, size=(2,2))
            #     imageStimulus.draw(); testWin.flip()

            # Show a prompt on grey background at the beginning of the trial and wait for a keypress.
            bg     = visual.GratingStim(win, tex=None, mask=None, size=2, units='norm', color=0)
            prompt = visual.TextStim(win, text=config.PROMPT_TEXT.format(trial * 100 // config.NTRIALS))
            # prompt = visual.TextStim(win, text='HELLO', font='Consolas')
            bg.draw(); prompt.draw(); win.flip(); event.waitKeys(clearEvents=True)
            # bg.draw(); win.flip(); event.waitKeys(clearEvents=True)
            # prompt.draw(); win.flip(); event.waitKeys(clearEvents=True)

            previousDigit = False

            # Start the cue loop.
            for cue in range(config.NCUES):

                # Create a random digit
                digit = randint(0, 9)

                # Ensure it is not the same as the previous digit
                while digit == previousDigit:
                    digit = randint(0, 9)

                # Set the new previous digit
                previousDigit = digit

                # Initialise the stimulus
                image    = config.IMAGES[digit]

                # Randomly choose a processor
                if random() < 0.5:
                    processor = 'control'
                    stimulus = config.PROCESSOR_CONTROL(digit, image, TRAINING_CONFIG.grid)
                else:
                    processor = 'test'
                    stimulus = config.PROCESSOR_TEST(digit, image, TRAINING_CONFIG.grid)

                # If this is a testing run, also draw the original image.
                if config.TESTING:
                    originalImage = visual.ImageStim(testWin, image=np.flipud(color.rgb2gray(image)), size=(2,2))
                    originalImage.draw(); testWin.flip()

                # Clear the event buffer
                event.clearEvents()

                # Set the mouse to the center. Might turn off, not sure which is better.
                mouse.setPos((0,0))

                # Initialise a False keypress
                keypressRaw = False

                # Set the cue clock to 0.
                clockCue.reset()

                # Set the mouse recording clock to 0
                mouseRecord.reset()

                if not config.WITH_SCANNING:
                    while not keypressRaw:
                        # Set the stimulus in the right half of the grid
                        stimulus.setPos(0.20, 0)
                        rendered = np.flipud(TRAINING_CONFIG.grid.render_values_array(stimulus.vector))
                        imstim = visual.ImageStim(win, image=rendered, size =  ( win.size[1] / (1.5 * win.size[0]), 1 / 1.5))
                        imstim.draw(); win.flip()

                        keypresses = event.waitKeys(keyList=config.KEY_LIST, clearEvents=True)
                        if keypresses:
                            keypressRaw = keypresses[0]
                else:
                    # Loop until the keypress
                    while not keypressRaw:

                        # Get the mouse position and set the stimulus to the position.
                        newPos = mouse.getPos()
                        newPos = [newPos[0], -newPos[1]]
                        stimulus.setPos(*newPos)

                        if mouseRecord.getTime() > config.MOUSE_RECORD_INTERVAL:

                            mouseRow = config.MOUSE_ROW_TEMPLATE.format(
                                trial=trial,
                                cue=cue,
                                digit=digit,
                                xmouse=newPos[0],
                                ymouse=newPos[1],
                                cuetime=clockCue.getTime(),
                                trialtime=clockTrial.getTime(),
                                sessiontime=clockSession.getTime(),
                            )
                            # mousefile.write(mouseRow)

                            mouseRecord.reset()

                        # Render the stimulus
                        rendered = np.flipud(TRAINING_CONFIG.grid.render_values_array(stimulus.vector))

                        # Create an image stimulus out of the rendered image.
                        # Then show the stimulus.
                        # Ensure stimulus is square on full screen window, assuming window has greater x dim than y dim.
                        imstim = visual.ImageStim(win, image=rendered, size =  ( win.size[1] / win.size[0], 1))
                        imstim.draw(); win.flip()

                        # Wait for a keypress.
                        # We only need the first keypress, and want the key input from the numpage.
                        keypresses = event.getKeys(keyList = config.KEY_LIST)
                        if keypresses:
                            keypressRaw = keypresses[0]
                        #keypressRaw, *_ = event.waitKeys(clearEvents=True, keyList=config.KEY_LIST)

                # Check if their input was correct.
                # Numpad keys are prepended with 'num_', so we strip it out.
                keypress = keypressRaw.strip('num_')
                correct  = (digit == int(keypress))

                # Create the data line.
                row = config.SESSION_ROW_TEMPLATE.format(
                    trial=trial,
                    cue=cue,
                    digit=digit,
                    processor=processor,
                    keypress=keypress,
                    cuetime=clockCue.getTime(),
                    trialtime=clockTrial.getTime(),
                    sessiontime=clockSession.getTime(),
                )

                # Write the data line to the session file.
                outfile.write(row)

                # Play the feedback sound.
                # correctSound.play() if correct else incorrectSound.play()

                # Play the digit sound.
                # digitSounds[digit].play()

        # At the end of all the trials, show an end screen and wait for key press
        # to exit.
        bg  = visual.GratingStim(win, tex=None, mask=None, size=2, units='norm', color=0)
        end = visual.TextStim(win, text=config.END_TEXT)
        bg.draw(); end.draw(); win.flip(); event.waitKeys(clearEvents=True)
        # bg.draw(); win.flip(); event.waitKeys(clearEvents=True)
