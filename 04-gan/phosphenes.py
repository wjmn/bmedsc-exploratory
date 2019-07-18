#!/usr/bin/env python

import numpy as np
import random
import tensorflow as tf
import keras
from math import e
from scipy.ndimage import gaussian_filter
from skimage import color
from abc import ABC, abstractmethod

def safebound(
        value: float,
        width: float,
        lower: float,
        upper: float
) -> float:
    """ 
    Returns the bounded min and max about value with width.
    """
    vmin = int(max(lower, value - width / 2))
    vmax = int(min(upper, value + width / 2))
    return vmin, vmax

def bound(
        value:float,
        lower: float,
        upper:float
) -> float:
    """
    Returns a bounded value.
    """
    if value > lower:
        if value < upper:
            return value
        return upper
    return lower


class Abstract:
    """
    Abstract base class for a grid.
    """

    def render_square(
            x : float,
            y : float,
            x_render : int,
            y_render : int,
            x_size : int,
            y_size : int,
            strength : float
    ) -> np.ndarray:
        """
        Converts a location (x and y domain [0, 1]) to a rendered square.
        """

        xmin, xmax = safebound(
            value = x * x_render,
            width = xsize,
            lower = 0,
            upper = x_render
        )

        ymin, ymax = safebound(
            value = y * y_render,
            width = ysize,
            lower = 0,
            upper = y_render
        )

        base = np.zeros((y_render, x_render))

        base[ymin:ymax, xmin:xmax] = strength        

        return base       

    def render_phosphene(
            x : float,
            y : float,
            x_render: int,
            y_render : int,
            x_size : int,
            y_size : int,
            strength : float
    ) -> np.ndarray:

        square = self.render_square(
            x = x,
            y = y,
            x_render = x_render,
            y_render = y_render,
            x_size = x_size,
            y_size = y_size,
            strength = strength
        )
       
        blurred = gaussian_filter(
            input = base,
            sigma = (self.xsize * self.ysize) ** 0.5,
            mode = 'constant'
        )

        blurred_max = blurred.max()
        
        if blurred_max > 0:
            scaling = self.strength / blurred_max
        else:
            scaling = 1
        
        blurred = blurred * scaling
        
        if blurred.max() > 1:
            return blurred / blurred.max()
        else: 
            return blurred

    def render_volume(
            locations : np.ndarray,
            sizes : np.ndarray,
            strengths : np.ndarray
    ) -> np.ndarray:
        
        volume = np.array([
                render_phosphene(
                    x = x / x_render,
                    y = y / y_render,
                    x_render = x_render,
                    y_render = y_render,
                    x_size = (x_render // x_phosphenes) // 2,
                    y_size = (y_render // y_phosphenes) // 2,
                    strength = 1,
                )
                for (
                    (x, y),
                    (x_size, y_size),
                    strength
                ) in zip(
                        self.locations,
                        self.sizes,
                        self.strengths
                    )
            ])

        return volume
 
    

class Cartesian(Abstract):
    """ 
    Create a Cartesian grid of phosphenes of REGULAR size.
    """

    def __init__(
            self,
            x_phosphenes : int,
            y_phosphenes : int,
            x_render : int,
            y_render : int,
            half : bool = False,
            random : bool = False
    ) -> Cartesian:

        """
        Args:
            x_phosphenes (int): the number of phosphenes in the x dimension.
            y_phosphenes (int): the number of phosphenes in the y dimension.
            x_render (int): the number of pixels in the x dimension of the render.
            y_render (int): the number of pixels in the y dimension of the render.
            half (bool): Whether to only fill the right-half of the render (default False).
            random (bool): Whether to randomise the locations of phosphenes (default False).
        """

        if random:
            self.locations = np.random.rand(
                x_phosphenes,
                y_phosphenes
            )
        else:
            self.locations = np.array([
                (x / x_render,
                 y / y_render)
                for x in range(x_phosphenes)
                for y in range(y_phosphenes)
            ])

        if half:
            self.locations[:, 0] = self.locations[:, 0] + 1 / 2

        self.sizes = np.array([
            ((x_render // x_phosphenes) // 2,
             (y_render // y_phosphenes) // 2)
            for x in range(x_phosphenes)
            for y in range(y_phosphenes)
        ])

        self.strengths = np.array([
            1
            for x in range(x_phosphenes)
            for y in range(y_phosphenes)
        ])

        self.volume = render_volume()
            

class Polar(Abstract):
    """
    Create a Polar grid of phosphenes, with size determined by location.
    """

    def __init__(
            self,
            n_theta : int,
            n_radius : int,
            x_render : int,
            y_render : int,
            half : bool = False,
            random : bool = False,
    ) -> Polar:
        
        """
        Args
            n_theta (int): number of angles in the grid.
            n_radius (int): number of radii in the grid.
            x_render (int): the number of pixels in the x dimension of the render.
            y_render (int): the number of pixels in the y dimension of the render.
            half (bool): whether to fill the right-half of the render (default False)
            random (bool): whether to randomise the locations of phosphenes (default False).
        """

        if random:
            # Purely random, same as the random locations for cartesian phosphenes.
            self.locations = np.random.rand(
                n_theta,
                n_radius
            )
            if half:
                self.locations[:, 0] = self.locations[:, 0] + 1 / 2
        else:
            if half:
                self.locations = np.array([
                    (0.5 + (i_radius / y_render * np.cos(self.i_angle(i_theta, n_theta))) / 2,
                     0.5 + (i_radius / y_render * np.sin(self.i_angle(i_theta, n_theta))) / 2,)
                    for i_radius in range(1, self.n_radius + 1)
                    for i_theta in range(self.n_theta )
                ])

    def i_angle(self, i_theta, n_theta):
        """
        Calculates the angle for angle of index i in range n
        """
        angle = (np.pi / (n_theta - 1) * i_theta) - (np.pi / 2)
        return angle
 
            
        
        


# Electrodes, which produce phosphenes.

class Electrode:
    """
    Produces a phosphene for a single electrode.
    """
    def __init__(self,
                 x: float,
                 y: float,
                 strength: float,
                 xdim: int,
                 ydim: int):
        """
        Args:
            x: float         - position in range [0, 1]. 
            y: float         - position in range [0, 1]
            strength: float  - relative brightness of the electrode in range [0, 1]
            xdim: int        - x dim of the output image
            ydim: int        - y dim of the output image 
        """
        self.x = x
        self.y = y
        self.strength = strength
        self.xdim = xdim
        self.ydim = ydim
        
        k = self.xdim / 2 + self.ydim / 2
        a = e * (self.xdim + self.ydim) / 128
        
        self.xsize = np.log(k * ((x-0.5)**2 + (y-0.5)**2) + a)
        self.ysize = np.log(k * ((x-0.5)**2 + (y-0.5)**2) + a)

        self.rendered = self.render()
        


    def render(self):
        xmin, xmax = safebound(self.xdim * self.x, self.xsize, 0, self.xdim)
        ymin, ymax = safebound(self.ydim * self.y, self.ysize, 0, self.ydim)

        base = np.zeros((self.ydim, self.xdim))
        base[ymin:ymax, xmin:xmax] = self.strength
        
        blurred = gaussian_filter(base, (self.xsize * self.ysize) ** 0.5, mode='constant')
        blurred_max = blurred.max()
        
        if blurred_max > 0:
            scaling = self.strength / blurred_max
        else:
            scaling = 1
        
        # Scale by offset back to previous ma
        blurred = blurred * scaling
        
        # Clip
        if blurred.max() > 1:
            return blurred / blurred.max()
        else: 
            return blurred

class RandomElectrode(Electrode):
    """
    This class introduced random distortions to the rendered phosphene.
    """
    def __init__(self,
                 xdim: int,
                 ydim: int):
        
        x = (random.random() + 1) / 2 # for hemisphere
        y = random.random()
        strength = 1
        
        Electrode.__init__(self, x, y, strength, xdim, ydim)

# Grids, which are composed of electrodes.

class Grid(ABC): 
    def __init__(self,
                 ndim1: int, 
                 ndim2: int, 
                 xdim: int, 
                 ydim: int):
        """
        Base class for a rendering grid.
        
        Args:
            ndim1: int - number of electrodes for dimension 1
            ndim2: int - number of electrodes for dimension 2
            xdim: int  - x dimension of output image
            ydim: int  - y dimension of output image
        """
        self.ndim1 = ndim1
        self.ndim2 = ndim2
        self.vector_size = ndim1 * ndim2
        self.xdim = xdim
        self.ydim = ydim
        
        self.grid = self.make_grid()
        self.prerendered = np.array([electrode.rendered for electrode in self.grid])
        self.prerendered_tensor = tf.convert_to_tensor(self.prerendered, dtype=tf.float32)
        
        super().__init__()
        
    @abstractmethod
    def make_grid(self):
        pass
    
    def show_locations(self):
        locations = np.array([electrode.location() for electrode in self.grid])
        summed = np.sum(locations, axis=0)
        return summed
    
    def render(self, values: np.ndarray):
        
        # Multiply the values with the renders and sum
        product = values.reshape(self.vector_size, 1, 1) * self.prerendered
        summed = sum(product)

        # Clip, then scale between -1 and 1
        clipped = np.clip(summed, 0, 1) * 2 - 1
        
        return clipped
    
    def render_tensor(self, tensor):
        
        # Preprocessing
        tiled = tf.tile(tensor, tf.constant([self.xdim]))
        reshaped = tf.reshape(tiled, (self.xdim, self.vector_size, 1))
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        
        # Multiply the values with the renders and sum
        product = transposed * self.prerendered_tensor
        summed = tf.reduce_sum(product, axis=0)
        
        # Clip, then scale by -1 and 1
        clipped = tf.clip_by_value(summed, 0, 1) * 2 - 1
        
        return clipped
        

class CartesianGrid(Grid):
    """
    A regular grid of electrodes with even spacin.
    """
    def __init__(self,
                 nxelectrode: int,
                 nyelectrode: int,
                 xdim: int,
                 ydim: int):
        """
        Args:
            nxelectrode: int - number of electrodes on x axis
            nyelectrode: int - number of electrodes on y axis
            xdim: int       - output x dimension of image
            ydim: int       - output y dimension of image
        """
        Grid.__init__(self, nxelectrode, nyelectrode, xdim, ydim)
        
    def make_grid(self):
        
        grid = [
            Electrode(x = x / self.ndim1,
                      y = y / self.ndim2,
                      strength = 1,
                      xdim = self.xdim,
                      ydim = self.ydim)
            for x in range(self.ndim1)
            for y in range(self.ndim2)
        ]
        
        return grid

class PolarGrid(Grid):
    """
    A polar regular grid of electrodes with even spacing 
    and size increasing with eccentricity. 
    """
    def __init__(self,
                 nradius: int,
                 ntheta: int,
                 xdim: int,
                 ydim: int):
        """
        Args:
            nradius: int - number of radii to place electrodes on
            ntheta: int  - number of angles to place electrodes on
            xdim: int    - output x dimension of image
            ydim: int    - output y dimension of image
        """
        Grid.__init__(self, nradius, ntheta, xdim, ydim)
        
    def iangle(self, i):
        """
        Calculates the angle for angle of index i in range(self.ndim2)
        """
        angle = (np.pi / (self.ndim2 - 1) * i) - (np.pi / 2)
        return angle
        
    def make_grid(self):
        
        xys = [
            (0.5 + (ir / self.ndim1 * np.cos(self.iangle(itheta))) / 2,
             0.5 + (ir / self.ndim1 * np.sin(self.iangle(itheta))) / 2,)
            for ir in range(1, self.ndim1 + 1)
            for itheta in range(self.ndim2)
        ]
        
        grid = [
            Electrode(x = x,
                      y = y,
                      strength = 1,
                      xdim = self.xdim,
                      ydim = self.ydim)
            for (x, y) in xys
        ]
        
        return grid
    
class RandomPolarGrid(PolarGrid):
    """
    A polar grid with random electrodes.
    """
        
    def make_grid(self):

        grid = [
            RandomElectrode(xdim = self.xdim,
                            ydim = self.ydim)
            for _ in range(self.ndim1)
            for _ in range(self.ndim2)
        ]
        
        return grid
    
class RescalingRandomPolarGrid(RandomPolarGrid):
    """
    A polar grid with random electrodes and non-summative rendering
    (rendering rescales the brightness to max). 
    """
    
    def render(self, values):
        
        # Multiply the values with the renders and sum
        product = values.reshape(self.vector_size, 1, 1) * self.prerendered
        summed = sum(product)
        summax = np.max(summed)

        # Rescale
        scaled = (summed / summax )
        
        # Clip below 0.5, then scale between -1 and 1
        clipped = np.clip(scaled, 0.5, 1) * 4 - 3 
        
        return clipped
    
    def render_tensor(self, tensor):
        
        # Preprocessing
        tiled = tf.tile(tensor, tf.constant([self.xdim]))
        reshaped = tf.reshape(tiled, (self.xdim, self.vector_size, 1))
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        
        # Multiply the values with the renders and sum
        product = transposed * self.prerendered_tensor
        summed = tf.reduce_sum(product, axis=0)
        summax = tf.reduce_max(summed)
        
        # Rescale
        scaled = tf.divide(summed, summax ) 
        
        # Clip below 0.5, then scale between -1 and 1
        clipped = tf.clip_by_value(scaled, 0.5, 1) * 4 - 3
        
        return clipped
        
# STIMULUS

class Stimulus:
    def __init__(self, image, grid, xpos=0, ypos=0):
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
        
        self.xpos = xpos
        self.ypos = ypos
        
        self.image = self.getImage()
        
        self.grid = grid
        self.sampleWidth = 2
        
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

        params = np.array([self.get_params(e.x, e.y) for e in self.grid.grid])
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

    def __init__(self, image, grid, encoder_path):
        self.encoder = tf.keras.models.load_model(encoder_path)      
        Stimulus.__init__(self, image, grid)
    
    def process(self):
        image_tensor = tf.convert_to_tensor(np.array([self.image]), dtype=tf.float32)
        return self.encoder(image_tensor).numpy()[0]
