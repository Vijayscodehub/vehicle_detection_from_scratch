import unittest

import torch

from src.geometry import box_iou, decode, detection_loss, encode, make_anchors, match, nms
from src.model import VehicleDetector


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_iou_known_examples(self):
        boxes = torch.tensor([[0., 0., 1., 1.], [1., 1., 2., 2.]])
        self.assertTrue(torch.equal(box_iou(boxes, boxes), torch.eye(2)))
        overlap = box_iou(boxes[:1], torch.tensor([[0., 0., 0.5, 1.]]))
        self.assertAlmostEqual(float(overlap[0, 0]), 0.5)
        self.assertEqual(box_iou(boxes, torch.empty(0, 4)).shape, (2, 0))

    def test_round_trip(self):
        anchors = make_anchors()[:2]
        boxes = torch.tensor([[0.1, 0.2, 0.4, 0.5], [0.5, 0.5, 0.8, 0.9]])
        self.assertTrue(torch.allclose(decode(encode(boxes, anchors), anchors), boxes, atol=1e-6))

    def test_matching_thresholds_and_forced_unique_anchors(self):
        anchors = torch.tensor([[0., 0., 1., 1.], [0., 0., 0.6, 1.],
                                [0., 0., 0.45, 1.], [2., 2., 3., 3.]])
        labels, _ = match(anchors, torch.tensor([[0., 0., 1., 1.]]))
        self.assertEqual(labels.tolist(), [1, 1, -1, 0])
        targets = torch.tensor([[0., 0., 0.01, 0.01], [0., 0., 0.02, 0.02]])
        labels, _ = match(make_anchors(), targets)
        self.assertGreaterEqual(int((labels == 1).sum()), 2)
        labels, offsets = match(make_anchors(), torch.empty(0, 4))
        self.assertEqual(int(labels.sum()), 0)
        self.assertEqual(float(offsets.sum()), 0)

    def test_nms(self):
        boxes = torch.tensor([[0., 0., 1., 1.], [0.01, 0.01, 1., 1.], [2., 2., 3., 3.]])
        self.assertEqual(nms(boxes, torch.tensor([0.9, 0.8, 0.7])).tolist(), [0, 2])
        self.assertEqual(nms(torch.empty(0, 4), torch.empty(0)).tolist(), [])

    def test_cpu_forward_loss_backward(self):
        model = VehicleDetector().cpu()
        logits, offsets = model(torch.zeros(2, 3, 320, 320))
        self.assertEqual(logits.shape, (2, 4500, 2))
        self.assertEqual(offsets.shape, (2, 4500, 4))
        targets = [torch.tensor([[0.2, 0.2, 0.4, 0.4]]), torch.empty(0, 4)]
        loss = detection_loss(logits, offsets, model.anchors, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()