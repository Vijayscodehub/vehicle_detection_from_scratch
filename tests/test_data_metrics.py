import unittest
from unittest.mock import patch

import torch
from PIL import Image

from src.data import VehicleDataset, horizontal_flip, letterbox
from src.metrics import evaluate_predictions


class DataMetricTests(unittest.TestCase):
    def test_letterbox_and_flip(self):
        image = Image.new("RGB", (1280, 720))
        boxes = torch.tensor([[100., 100., 300., 200.]])
        resized, transformed = letterbox(image, boxes)
        self.assertEqual(resized.size, (320, 320))
        expected = torch.tensor([[25., 95., 75., 120.]]) / 320
        self.assertTrue(torch.allclose(transformed, expected))
        _, flipped = horizontal_flip(image, boxes)
        self.assertEqual(flipped.tolist(), [[980., 100., 1180., 200.]])
        _, restored = horizontal_flip(image, flipped)
        self.assertTrue(torch.equal(restored, boxes))
        _, empty = letterbox(image, torch.empty(0, 4))
        self.assertEqual(empty.shape, (0, 4))

    def test_pending_dataset_is_rejected(self):
        document = {"images": [{"source_id": "pending_image", "qa_status": "pending"}]}
        with patch("src.data.json.loads", return_value=document):
            with self.assertRaisesRegex(ValueError, "unreviewed"):
                VehicleDataset("train")

    def test_partial_pilot_mode_is_training_only(self):
        for split in ("validation", "test"):
            with self.assertRaisesRegex(ValueError, "training-only"):
                VehicleDataset(split, pilot_only=True)

    def test_actual_reviewed_pilot_loader(self):
        dataset = VehicleDataset("train", pilot_only=True)
        self.assertGreater(len(dataset), 0)
        self.assertTrue(all(image["qa_status"] == "reviewed" for image in dataset.images))
        image, boxes, _ = dataset[0]
        self.assertEqual(image.shape, (3, 320, 320))
        self.assertGreater(len(boxes), 0)

    def test_perfect_and_duplicate_predictions(self):
        box = torch.tensor([[0., 0., 1., 1.]])
        predictions = {"one": (box.repeat(2, 1), torch.tensor([0.9, 0.8]))}
        result = evaluate_predictions(predictions, {"one": box})
        self.assertEqual(result["ap50"], 1.)
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["missed_targets"], 0)

    def test_missed_targets_and_wrong_boxes(self):
        targets = {"one": torch.tensor([[0., 0., 0.2, 0.2]])}
        predictions = {"one": (torch.tensor([[0.7, 0.7, 1., 1.]]), torch.tensor([0.9]))}
        result = evaluate_predictions(predictions, targets)
        self.assertEqual(result["ap50"], 0.)
        self.assertEqual(result["missed_targets"], 1)
        self.assertEqual(result["false_positives"], 1)
        result = evaluate_predictions({"one": (torch.empty(0, 4), torch.empty(0))}, targets)
        self.assertEqual(result["ap50"], 0.)

    def test_false_positive_before_true_positive(self):
        boxes = torch.tensor([[0.7, 0.7, 1., 1.], [0., 0., 0.2, 0.2]])
        result = evaluate_predictions({"one": (boxes, torch.tensor([0.9, 0.8]))},
                                      {"one": boxes[1:]})
        self.assertEqual(result["ap50"], 0.5)


if __name__ == "__main__":
    unittest.main()