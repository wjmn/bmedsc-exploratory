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


class AbstractGrid:
    """
    Abstract base class for a grid.
    """

    def render_square(
            self,
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
            width = x_size,
            lower = 0,
            upper = x_render
        )

        ymin, ymax = safebound(
            value = y * y_render,
            width = y_size,
            lower = 0,
            upper = y_render
        )

        base = np.zeros((y_render, x_render))

        base[ymin:ymax, xmin:xmax] = strength        

        return base       

    def render_phosphene(
            self,
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
            input = square,
            sigma = (x_size * y_size) ** 0.5,
            mode = 'constant'
        )

        blurred_max = blurred.max()
        
        if blurred_max > 0:
            scaling = strength / blurred_max
        else:
            scaling = 1
        
        blurred = blurred * scaling
        
        if blurred.max() > 1:
            return blurred / blurred.max()
        else: 
            return blurred

    def render_volume(self, x_render : int, y_render : int) -> np.ndarray:
        
        volume = np.array([
                self.render_phosphene(
                    x = x,
                    y = y,
                    x_render = x_render,
                    y_render = y_render,
                    x_size = x_size,
                    y_size = y_size,
                    strength = strength,
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
 
    

class CartesianGrid(AbstractGrid):
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
    ):

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
                x_phosphenes * y_phosphenes,
                2,
            )
        else:
            self.locations = np.array([
                (x / (x_phosphenes - 1),
                 y / (y_phosphenes - 1))
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

        self.volume = self.render_volume(
            x_render,
            y_render,
        )
            

class PolarGrid(AbstractGrid):
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
    ):
        
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
                n_theta * n_radius,
                2
            )
            if half:
                self.locations[:, 0] = self.locations[:, 0] + 1 / 2
        else:
            self.locations = np.array([
                (0.5 + (i_radius / n_radius * np.cos(self.i_angle(i_theta, n_theta, half))) / 2,
                 0.5 + (i_radius / n_radius * np.sin(self.i_angle(i_theta, n_theta, half))) / 2,)
                for i_radius in range(1, n_radius + 1)
                for i_theta in range(n_theta)
            ])

        k = x_render / 2 + y_render / 2
        a = e * (x_render + y_render) / 128

        self.sizes = np.array([
            (np.log(k * ((x-0.5)**2 + (y-0.5)**2) + a),
             np.log(k * ((x-0.5)**2 + (y-0.5)**2) + a))
            for (x, y) in self.locations
        ])

        self.strengths = np.array([
            1
            for _ in range(n_theta)
            for _ in range(n_radius)
        ])

        self.volume = self.render_volume(
            x_render,
            y_render,
        )

    def i_angle(self, i_theta, n_theta, half=False):
        """
        Calculates the angle for angle of index i in range n
        """
        
        if half:
            angle = (np.pi / (n_theta - 1) * i_theta) - (np.pi / 2)

        else:
            angle = 2 * np.pi / n_theta * i_theta

        return angle
        
def render(grid : AbstractGrid, values : np.ndarray) -> np.ndarray:

    # Multiply the values with the renders and sum
    product = values.reshape(len(grid.locations), 1, 1) * grid.volume
    summed = sum(product)

    # Clip, then scale between -1 and 1
    clipped = np.clip(summed, 0, 1) * 2 - 1

    return clipped

def render_tensor(grid : AbstractGrid, values : tf.Tensor) -> np.ndarray:

    # Preprocessing
    tiled = tf.tile(values, tf.constant([grid.volume.shape[1]]))
    reshaped = tf.reshape(tiled, (grid.volume.shape[1], len(grid.locations), 1))
    transposed = tf.transpose(reshaped, perm=[1, 0, 2])

    # Multiply the values with the renders and sum
    product = transposed * tf.constant(grid.volume)
    summed = tf.reduce_sum(product, axis=0)

    # Clip, then scale by -1 and 1
    clipped = tf.clip_by_value(summed, 0, 1) * 2 - 1

    return clipped
