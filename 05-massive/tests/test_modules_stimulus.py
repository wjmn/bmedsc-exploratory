import unittest
import modules.stimulus


class BoundCase(unittest.TestCase):

    def test_value_lower(self):
        self.assertEqual(
            modules.stimulus.value_bound(-0.2, 0, 1),
            0
        )

    def test_value_upper(self):
        self.assertEqual(
            modules.stimulus.value_bound(1.3, 0, 1),
            1
        )

    def test_value_within(self):
        self.assertEqual(
            modules.stimulus.value_bound(0.5, 0, 1),
            0.5
        )


def main():
    unittest.main()


if __name__ == '__main__':
    main()
