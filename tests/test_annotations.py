import unittest

from src.annotations import validate_boxes


class AnnotationTests(unittest.TestCase):
    def test_valid_box_and_empty_image(self):
        validate_boxes([[0, 0, 20, 20], [100, 200, 300, 400]], 1280, 720)
        validate_boxes([], 1280, 720)

    def test_invalid_boxes(self):
        for box in ([1, 2, 3], [0, 0, 19, 20], [-1, 2, 40, 50],
                    [0, 0, 1300, 50], [30, 30, 10, 10]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                validate_boxes([box], 1280, 720)


if __name__ == "__main__":
    unittest.main()