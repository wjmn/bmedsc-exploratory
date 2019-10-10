""" Handles operations on grid including creation and rendering.
"""

import numpy as np
import tensorflow as tf
import pickle
from datetime import datetime
from enum import Enum
from math import e
from scipy.ndimage import gaussian_filter
from typing import Tuple
from math import floor, ceil


# Utility functions
def width_bound(
        value: float,
        width: float,
        lower: float,
        upper: float
) -> Tuple[int, int]:
    """ Returns the integer, bounded min and max about value with width.
    """
    v_min = floor(max(lower, value - width / 2))
    v_max = floor(min(upper, value + width / 2))
    return v_min, v_max


# Grid type enumeration
class GridType(Enum):
    CARTESIAN = 0
    POLAR = 1


# Grid classes
class AbstractGrid:
    """ Abstract base class for a grid.
    """

    grid_id: str
    grid_type: GridType
    num_phosphenes: int
    render_shape: Tuple[int, int]
    half: bool
    random: bool
    locations: np.ndarray
    sizes: np.ndarray
    strengths: np.ndarray
    volume: np.ndarray

    def _set_id(self) -> None:
        self.grid_id = 'G-' + datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S-%f')

    def prerender_square(
            self,
            x: float,
            y: float,
            x_render: int,
            y_render: int,
            x_size: int,
            y_size: int,
            strength: float
    ) -> np.ndarray:
        """ Converts a location (x and y domain [0, 1]) to a rendered square.
        """

        x_min, x_max = width_bound(
            value=x*x_render,
            width=x_size,
            lower=0,
            upper=x_render
        )

        y_min, y_max = width_bound(
            value=y*y_render,
            width=y_size,
            lower=0,
            upper=y_render
        )

        base = np.zeros((y_render, x_render))
        base[y_min:y_max, x_min:x_max] = strength
        return base

    def prerender_phosphene(
            self,
            x: float,
            y: float,
            x_render: int,
            y_render: int,
            x_size: int,
            y_size: int,
            strength: float
    ) -> np.ndarray:
        """ Render a single phosphene.
        """

        square = self.prerender_square(
            x=x,
            y=y,
            x_render=x_render,
            y_render=y_render,
            x_size=x_size,
            y_size=y_size,
            strength=strength
        )

        blurred = gaussian_filter(
            input=square,
            sigma=(x_size * y_size) ** 0.5,
            mode='constant'
        )

        if blurred.max() > 0:
            scaling = strength / blurred.max()
            blurred = blurred * scaling

        if blurred.max() > 1:
            return blurred / blurred.max()

        return blurred

    def prerender_volume(
            self,
            x_render: int,
            y_render: int
    ) -> np.ndarray:
        """ Pre-render the entire grid as a 3D volume.
        """

        volume = np.array([
            self.prerender_phosphene(
                x=x,
                y=y,
                x_render=x_render,
                y_render=y_render,
                x_size=x_size,
                y_size=y_size,
                strength=strength,
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

    def render_values_array(
            self,
            values: np.ndarray
    ) -> np.ndarray:
        """ Renders a grid with the given values for each phosphene.

        :param values: An array of values for each phosphene.
        :return: A rendered 2D array of greyscale values in range -1 to 1.
        """
        # Multiply the values with the renders and sum
        product = values.reshape((self.num_phosphenes, 1, 1)) * self.volume
        summed = sum(product)

        # Clip, then scale between -1 and 1
        clipped = np.clip(summed, 0, 1) * 2 - 1

        return clipped

    def render_values_tensor(
            self,
            values: tf.Tensor
    ) -> tf.Tensor:
        """ Renders a grid with given tensor values for each phosphene.

        :param values: A 1D tensorflow vector ("tensor") of values for each phosphene.
        :return: A rendered 2D tensor of greyscale values in range -1 to 1.
        """
        # Pre-processing
        tiled = tf.tile(values, [self.render_shape[0]])
        reshaped = tf.reshape(tiled, (self.render_shape[0], self.num_phosphenes, 1))
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])

        # Multiply the values with the renders and sum
        product = transposed * self.volume
        summed = tf.reduce_sum(product, axis=0)

        # Clip, then scale by -1 and 1
        clipped = tf.clip_by_value(summed, 0, 1) * 2 - 1

        return clipped

    def save(self, data_dir: str) -> None:
        """ Save a grid to a specified file path as a pickle file.

        :param data_dir: grid directory with slash.
        """

        file_path = data_dir + f'{self.grid_id}.pkl'

        with open(file_path, 'wb') as outfile:
            pickle.dump(self, outfile)


class CartesianGrid(AbstractGrid):

    def __init__(
            self,
            x_phosphenes: int,
            y_phosphenes: int,
            x_render: int,
            y_render: int,
            half: bool = False,
            random: bool = False
    ) -> None:
        """ Create a cartesian grid of phosphenes (regularly sized).

        :param x_phosphenes: the number of phosphenes in the x dimension.
        :param y_phosphenes: the number of phosphenes in the y dimension.
        :param x_render: the number of pixels in the x dimension of the render.
        :param y_render: the number of pixels in the y dimension of the render.
        :param half: whether to only fill the right-half of the render (default False).
        :param random: Whether to randomise the locations of phosphenes (default False).
        """

        self._set_id()
        self.grid_type = GridType.CARTESIAN
        self.num_phosphenes = x_phosphenes * y_phosphenes
        self.render_shape = (x_render, y_render)
        self.half = half
        self.random = random

        if random:
            self.locations = np.random.rand(x_phosphenes * y_phosphenes, 2)
        else:
            self.locations = np.array([
                (x / (x_phosphenes - 1), y / (y_phosphenes - 1))
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

        self.volume = self.prerender_volume(
            x_render,
            y_render,
        )


class PolarGrid(AbstractGrid):

    def __init__(
            self,
            n_theta: int,
            n_radius: int,
            x_render: int,
            y_render: int,
            half: bool = False,
            random: bool = False,
    ) -> None:
        """ Create a polar grid of phosphenes (size varies with eccentricity).

        :param n_theta: number of angles in the grid.
        :param n_radius: number of radii in the grid.
        :param x_render: the number of pixels in the x dimension of the render.
        :param y_render: the number of pixels in the y dimension of the render.
        :param half: whether to fill the right-half of the render (default False)
        :param random: whether to randomise the locations of phosphenes (default False).
        """

        self._set_id()
        self.grid_type = GridType.POLAR
        self.num_phosphenes = n_theta * n_radius
        self.render_shape = (x_render, y_render)
        self.half = half
        self.random = random

        if random:
            self.locations = np.random.rand( n_theta * n_radius, 2 )
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
        a = e * (x_render + y_render) / 32

        self.sizes = np.array([
            (np.log(k * ((x - 0.5) ** 2 + (y - 0.5) ** 2) + a),
             np.log(k * ((x - 0.5) ** 2 + (y - 0.5) ** 2) + a))
            for (x, y) in self.locations
        ])

        self.strengths = np.array([
            1
            for _ in range(n_theta)
            for _ in range(n_radius)
        ])

        self.volume = self.prerender_volume(
            x_render,
            y_render,
        )

    def i_angle(self, i_theta: float, n_theta: float, half: bool = False) -> float:
        """ Calculates the angle for angle of index i in range n.
        """
        if half:
            angle = (np.pi / (n_theta - 1) * i_theta) - (np.pi / 2)
        else:
            angle = 2 * np.pi / n_theta * i_theta
        return angle


# Grid loader
def load(file_path: str) -> AbstractGrid:
    """ Load a grid from a file path.

    :param file_path:
    """
    with open(file_path, 'rb') as infile:
        grid = pickle.load(infile)
    return grid
