import json
import unittest

import torch

from src.geometry import detection_loss
from src.model import VehicleDetector
from src.prepare_data import ROOT
from src.train import seed_everything


class TrainingTests(unittest.TestCase):
    def test_seed_repeats_initial_weights_and_loss(self):
        losses = []
        for _ in range(2):
            seed_everything()
            model = VehicleDetector().cpu()
            logits, offsets = model(torch.zeros(1, 3, 320, 320))
            loss = detection_loss(logits, offsets, model.anchors,
                                  [torch.tensor([[0.2, 0.2, 0.4, 0.4]])])
            losses.append(float(loss.detach()))
        self.assertEqual(losses[0], losses[1])

    def test_ablation_changes_only_augmentation(self):
        baseline = json.loads((ROOT / "configs/baseline.json").read_text())
        augmented = json.loads((ROOT / "configs/augmentation.json").read_text())
        differences = {key for key in baseline if baseline[key] != augmented[key]}
        self.assertEqual(differences, {"name", "augment"})


if __name__ == "__main__":
    unittest.main()