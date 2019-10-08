import unittest
import modules.grid
import numpy as np
import tensorflow as tf


class BoundCase(unittest.TestCase):

    def test_width_both(self):
        self.assertEqual(
            modules.grid.width_bound(1, 4, 0, 2),
            (0, 2)
        )

    def test_width_lower(self):
        self.assertEqual(
            modules.grid.width_bound(2, 8, 0, 8),
            (0, 6)
        )

    def test_width_upper(self):
        self.assertEqual(
            modules.grid.width_bound(6, 8, 0, 8),
            (2, 8)
        )


class CartesianCase(unittest.TestCase):

    def setUp(self):
        self.X_PHOSPHENES = 8
        self.Y_PHOSPHENES = 8
        self.X_RENDER = 48
        self.Y_RENDER = 48
        self.HALF = False
        self.RANDOM = True

        self.grid = modules.grid.CartesianGrid(
            self.X_PHOSPHENES,
            self.Y_PHOSPHENES,
            self.X_RENDER,
            self.Y_RENDER,
            self.HALF,
            self.RANDOM
        )

        self.GRID_ID = self.grid.grid_id

    def test_render_array(self):
        values = np.random.rand(self.grid.num_phosphenes)
        self.grid.render_values_array(values)

    def test_render_tensor(self):
        values = tf.random.uniform((self.grid.num_phosphenes,))
        self.grid.render_values_tensor(values)

    def test_equal_renders(self):
        values = np.random.rand(self.grid.num_phosphenes)
        array_render = self.grid.render_values_array(values)
        tensor_render = self.grid.render_values_tensor(tf.convert_to_tensor(values, tf.float32))

        self.assert_(
            (array_render.shape == tensor_render.numpy().shape)
        )

        np.testing.assert_almost_equal(
            array_render,
            tensor_render,
            decimal=4
        )

    def test_load(self):
        self.grid.save('./test-saves/')

        new_grid = modules.grid.load(f'./test-saves/{self.GRID_ID}.pkl')

        self.assertEqual(
            self.grid.grid_id,
            new_grid.grid_id
        )


class PolarCase(unittest.TestCase):

    def setUp(self):
        self.N_THETA = 8
        self.N_RADIUS = 8
        self.X_RENDER = 48
        self.Y_RENDER = 48
        self.HALF = True
        self.RANDOM = False

        self.grid = modules.grid.CartesianGrid(
            self.N_THETA,
            self.N_RADIUS,
            self.X_RENDER,
            self.Y_RENDER,
            self.HALF,
            self.RANDOM
        )

        self.GRID_ID = self.grid.grid_id

    def test_render_array(self):
        values = np.random.rand(self.grid.num_phosphenes)
        self.grid.render_values_array(values)

    def test_render_tensor(self):
        values = tf.random.uniform((self.grid.num_phosphenes,))
        self.grid.render_values_tensor(values)

    def test_equal_renders(self):
        values = np.random.rand(self.grid.num_phosphenes)
        array_render = self.grid.render_values_array(values)
        tensor_render = self.grid.render_values_tensor(tf.convert_to_tensor(values, tf.float32))

        self.assert_(
            (array_render.shape == tensor_render.numpy().shape)
        )

        np.testing.assert_almost_equal(
            array_render,
            tensor_render,
            decimal=4
        )

    def test_load(self):

        self.grid.save('./test-saves/')

        new_grid = modules.grid.load(f'./test-saves/{self.GRID_ID}.pkl')

        self.assertEqual(
            self.grid.grid_id,
            new_grid.grid_id
        )


def main():
    unittest.main()


if __name__ == '__main__':
    main()
