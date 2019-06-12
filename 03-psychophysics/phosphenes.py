#!/usr/bin/env python

import numpy as np
from scipy.ndimage import gaussian_filter
import random
import math
from skimage import color

# CONSTANTS

XSIZE = 64
YSIZE = 64
PBASE = 2
SCALE = 6
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

# Electrodes, which produce phosphenes.

class Electrode:
    def __init__(self, x: float, y: float, xsize : int = XSIZE, ysize : int = YSIZE, randomPos: float = 0):
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
        self.xsize = xsize
        self.ysize = ysize

        self.size = PBASE * (0.5 + (4 * np.sqrt((self.x - 0.5) ** 2 + (self.y - 0.5) ** 2)) ** 2)

        self.rendered = self.render()

    def render(self):
        xmin, xmax = safebound(self.xsize * self.x, self.size, 0, self.xsize)
        ymin, ymax = safebound(self.ysize * self.y, self.size, 0, self.ysize)

        base = np.zeros((self.ysize, self.xsize))
        base[ymin:ymax, xmin:xmax] = 1

        return gaussian_filter(base, self.size)

class UniqueElectrode:
    """
    This class implements electrodes with unique characteristics such as colour and shape.
    """
    def __init__(self, x: float, y: float, xsize : int = XSIZE, ysize : int = YSIZE, randomPos: float = 0.001):
        self.x = bound(x + (random.random() - 0.5) * randomPos, 0, 1)
        self.y = bound(y + (random.random() - 0.5) * randomPos, 0, 1)
        self.size = PBASE * (0.5 + (4 * np.sqrt((self.x - 0.5) ** 2 + (self.y - 0.5) ** 2)) ** 2)
        #self.colour = np.random.random(3)
        self.brightness = np.random.random()
        # xmod and ymod modify the shape of the phosphene
        self.xmod = 1 + (random.random()-0.5) * 3
        self.ymod = 1 + (random.random()-0.5) * 3
        self.xsize = xsize
        self.ysize = ysize

        self.rendered = self.render()

    def render(self):
        xmin, xmax = safebound(self.xsize * self.x, self.size*self.xmod, 0, self.xsize)
        ymin, ymax = safebound(self.ysize * self.y, self.size*self.ymod, 0, self.ysize)

        # base = np.zeros((self.ysize, self.xsize, 3))
        # base[ymin:ymax, xmin:xmax, :] = self.colour
        base = np.zeros((self.ysize, self.xsize))
        base[ymin:ymax, xmin:xmax] = self.brightness
        # base = base.reshape((self.ysize, self.xsize))

        return gaussian_filter(base, self.size * (random.random() ** 0.3))

# Grids, which are composed of electrodes.

class RegularGrid:
    def __init__(self, exsize: int = EXSIZE, eysize: int = EYSIZE, xsize=XSIZE, ysize=YSIZE):
        """
        
        Args:
            exsize: int - x size of electrode grid 
            eysize: int - y size of electrode grid
        """
        self.exsize = EXSIZE
        self.eysize = EYSIZE
        self.grid = [
            Electrode(x / exsize, y / eysize, xsize=xsize, ysize=ysize)
            for x in range(exsize)
            for y in range(eysize)
        ]

    def render(self, values):
        product = [v * e.rendered for (v, e) in zip(values, self.grid)]
        summed = sum(product)
        summax = np.max(summed)
        return np.clip(summed, 0, 1) * 2 - 1
        # return (summed / summax) * 2 - 1

class IrregularGrid:
    def __init__(self, randomPos=2, exsize=EXSIZE, eysize=EYSIZE, xsize=XSIZE, ysize=YSIZE):
        self.exsize = EXSIZE
        self.eysize = EYSIZE
        self.grid = [
            Electrode(0.5 + (x / exsize) / 2, y / eysize, xsize=xsize, ysize=ysize, randomPos=randomPos )
            for x in range(exsize)
            for y in range(eysize)
        ]

    def render(self, values):
        product = [v * e.rendered for (v, e) in zip(values, self.grid)]
        summed = sum(product)
        summax = np.max(summed)
        return np.clip(summed, 0, 1) * 2 - 1
        # return (summed / summax) * 2 - 1

class PolarRegularGrid:
    def __init__(self, nrho, ntheta, xsize=XSIZE, ysize=YSIZE):
        self.nrho   = nrho
        self.ntheta = ntheta
        self.grid = [
            # Need to think of better way to scale.
            Electrode(((math.exp(rho**0.6) / math.exp(nrho**0.6) * math.cos((math.pi * theta / ntheta) - math.pi/2)) + 1) / 2,
                      ((math.exp(rho**0.6) / math.exp(nrho**0.6) * math.sin((math.pi * theta / ntheta) - math.pi/2)) + 1) / 2,
                      xsize = xsize,
                      ysize = ysize,
                     )
            # Ensure the central electrodes are actually visible by adding 1 to zero.
            for rho in range(1, nrho+1)
            for theta in range(ntheta)
        ]
        
        self.renders = tf.convert_to_tensor(np.array([e.rendered for e in self.grid]), dtype=tf.float32)

    def render(self, values):
        product = [v * e.rendered for (v, e) in zip(values, self.grid)]
        summed = sum(product)
        summax = np.max(summed)
        return np.clip(summed, 0, 1) * 2 - 1
        # return (summed / summax) * 2 - 1
        
    def render_tensor(self, tensor):
        reshaped = tf.transpose(tf.reshape(tf.tile(tensor, tf.constant([64])), (64, 144, 1)), perm=[1, 0, 2])
        product = reshaped * self.renders
        summed = tf.reduce_sum(product, axis=0)
        return tf.clip_by_value(summed, 0, 1) * 2 - 1

class PolarRegularUniqueGrid:
    def __init__(self, nrho, ntheta, xsize=XSIZE, ysize=YSIZE):
        self.nrho   = nrho
        self.ntheta = ntheta
        self.grid = [
            # Need to think of better way to scale.
            UniqueElectrode(((math.exp(rho**0.6) / math.exp(nrho**0.6) * math.cos((math.pi * theta / ntheta) - math.pi/2)) + 1) / 2,
                            ((math.exp(rho**0.6) / math.exp(nrho**0.6) * math.sin((math.pi * theta / ntheta) - math.pi/2)) + 1) / 2,
                            xsize = xsize,
                            ysize = ysize,
                           )
            # Ensure the central electrodes are actually visible by adding 1 to zero.
            for rho in range(1, nrho+1)
            for theta in range(ntheta)
        ]
        
        self.renders = tf.convert_to_tensor(np.array([e.rendered for e in self.grid]), dtype=tf.float32)

    def render(self, values):
        product = [v * e.rendered for (v, e) in zip(values, self.grid)]
        summed = sum(product)
        summax = np.max(summed)
        return np.clip(summed, 0, 1)
        # return (summed / summax) * 2 - 1
        
    def render_tensor(self, tensor):
        reshaped = tf.transpose(tf.reshape(tf.tile(tensor, tf.constant([64])), (64, 144, 1)), perm=[1, 0, 2])
        product = reshaped * self.renders
        summed = tf.reduce_sum(product, axis=0)
        return tf.clip_by_value(summed, 0, 1) * 2 - 1

class NonLinearInteractionGrid:
    def __init__(self, nrho, ntheta, xsize=XSIZE, ysize=YSIZE):
        self.nrho   = nrho
        self.ntheta = ntheta
        self.grid = [
            # Need to think of better way to scale.
            UniqueElectrode(((math.exp(rho**0.6) / math.exp(nrho**0.6) * math.cos((math.pi * theta / ntheta) - math.pi/2)) + 1) / 2,
                            ((math.exp(rho**0.6) / math.exp(nrho**0.6) * math.sin((math.pi * theta / ntheta) - math.pi/2)) + 1) / 2,
                            xsize = xsize,
                            ysize = ysize,
                           )
            # Ensure the central electrodes are actually visible by adding 1 to zero.
            for rho in range(1, nrho+1)
            for theta in range(ntheta)
        ]
        
        self.renders = tf.convert_to_tensor(np.array([e.rendered for e in self.grid]), dtype=tf.float32)

    def render(self, values):
        # Assume all inputs are in the range 0 and 1
        product = [v * e.rendered for (v, e) in zip(values, self.grid)]
        summed = sum(product)
        summax = np.max(summed)
        return (summed / summax) * 2 - 1
    
    def render_tensor(self, tensor):
        # Assume all inputs are in the range 0 and 1
        reshaped = tf.transpose(tf.reshape(tf.tile(tensor, tf.constant([64])), (64, 144, 1)), perm=[1, 0, 2])
        product = reshaped * self.renders
        summed = tf.reduce_sum(product, axis=0)
        summax = tf.reduce_max(summed)
        return tf.divide(summed, summax) * 2 - 1
        
        
# STIMULUS

class Stimulus:
    def __init__(self, image, grid, xpos=0, ypos=0):
        self.shape = image.shape
        
        if len(self.shape) == 2:
            self.original = image.reshape(*self.shape, 1)
            self.shape = self.original.shape
        else:
            self.original = image
        
        self.padder = np.zeros((3 * self.shape[0], 3 * self.shape[1], self.shape[2]))
        self.padder[self.shape[0]:2*self.shape[0], self.shape[1]:2*self.shape[1], :] = self.original
        
        self.xpos = xpos
        self.ypos = ypos
        
        self.image = self.getImage()
        
        self.grid = grid
        self.sampleWidth = 6
        
        self.vector = self.process()
            
    def get_params(self, x : float, y : float):
        
        ymin = bound(int(self.shape[0] * y - self.sampleWidth // 2), 0, self.shape[0] - 1)
        ymax = bound(int(self.shape[0] * y + self.sampleWidth // 2), 0, self.shape[0] - 1)
        xmin = bound(int(self.shape[1] * x - self.sampleWidth // 2), 0, self.shape[1] - 1)            
        xmax = bound(int(self.shape[1] * x + self.sampleWidth // 2), 0, self.shape[1] - 1)

        vals  = self.image[ymin:ymax, xmin:xmax, :]
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

        params = [self.get_params(e.x, e.y) for e in self.grid.grid]
        return params
        #flattened = self.image.flatten(order="C")
    
    def setPos(self, xpos: float, ypos: float):
        """Translate the image. xpos and ypos lie in the range (-1, 1)
        """
        self.xpos = xpos
        self.ypos = ypos
        self.image = self.getImage()
        self.vector = self.process()
        
        
# TESTING ONLY 

import tensorflow as tf
tf.executing_eagerly()
import keras

# input_shape = (72, 72)

# encoder_path = "./data/models/encoder_model.h5"
# encoder = tf.keras.models.load_model(encoder_path)

# encoder = make_encoder_model()

class StimulusNet(Stimulus):

    def __init__(self, image, grid, encoder_path):
        self.encoder = tf.keras.models.load_model(encoder_path)
        Stimulus.__init__(self, image, grid)
    
    def process(self):
        image_tensor = tf.convert_to_tensor(np.array([self.image]), dtype=tf.float32)
        print(image_tensor.shape)
        return self.encoder(image_tensor).numpy()[0]
