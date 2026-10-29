import unittest
from unittest.mock import patch

from src.annotations import export, validate_boxes


class AnnotationTests(unittest.TestCase):
    def test_valid_box_and_empty_image(self):
        validate_boxes([[0, 0, 20, 20], [100, 200, 300, 400]], 1280, 720)
        validate_boxes([], 1280, 720)

    def test_invalid_boxes(self):
        for box in ([1, 2, 3], [0, 0, 19, 20], [-1, 2, 40, 50],
                    [0, 0, 1300, 50], [30, 30, 10, 10]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                validate_boxes([box], 1280, 720)

    def test_require_complete_rejects_pending_without_writing(self):
        row = {"image_id": "pending", "width": "1280", "height": "720"}
        with patch("src.annotations.manifest", return_value=[row]):
            with patch("src.annotations.read_labels", return_value={}):
                with self.assertRaisesRegex(ValueError, "lack reviewed labels"):
                    export(require_complete=True)


if __name__ == "__main__":
    unittest.main()