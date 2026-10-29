"""Small, testable box operations. Coordinates are normalized xyxy unless noted."""

import math

import torch
from torch.nn import functional as functional


def box_iou(first, second):
    minimum = torch.maximum(first[:, None, :2], second[None, :, :2])
    maximum = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = (maximum - minimum).clamp(min=0).prod(dim=2)
    first_area = (first[:, 2:] - first[:, :2]).clamp(min=0).prod(dim=1)
    second_area = (second[:, 2:] - second[:, :2]).clamp(min=0).prod(dim=1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return intersection / union.clamp(min=1e-8)


def centers_and_sizes(boxes):
    return (boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]


def make_anchors():
    anchors = []
    for grid, scales in ((20, (0.06, 0.12, 0.24)), (10, (0.24, 0.40, 0.65))):
        for row in range(grid):
            for column in range(grid):
                center_x = (column + 0.5) / grid
                center_y = (row + 0.5) / grid
                for scale in scales:
                    for ratio in (0.5, 1.0, 2.0):
                        width = scale * math.sqrt(ratio)
                        height = scale / math.sqrt(ratio)
                        anchors.append([center_x - width / 2, center_y - height / 2,
                                        center_x + width / 2, center_y + height / 2])
    return torch.tensor(anchors, dtype=torch.float32)


def encode(boxes, anchors):
    box_center, box_size = centers_and_sizes(boxes)
    anchor_center, anchor_size = centers_and_sizes(anchors)
    center_offsets = (box_center - anchor_center) / anchor_size / 0.1
    size_offsets = torch.log(box_size.clamp(min=1e-8) / anchor_size) / 0.2
    return torch.cat((center_offsets, size_offsets), dim=1)


def decode(offsets, anchors):
    anchor_center, anchor_size = centers_and_sizes(anchors)
    center = anchor_center + offsets[:, :2] * 0.1 * anchor_size
    size = anchor_size * torch.exp((offsets[:, 2:] * 0.2).clamp(-8, 8))
    return torch.cat((center - size / 2, center + size / 2), dim=1)


def match(anchors, boxes):
    labels = torch.zeros(len(anchors), dtype=torch.long)
    offsets = torch.zeros_like(anchors)
    if len(boxes) == 0:
        return labels, offsets
    overlaps = box_iou(anchors, boxes)
    best_iou, matched = overlaps.max(dim=1)
    labels[(best_iou >= 0.4) & (best_iou < 0.5)] = -1
    labels[best_iou >= 0.5] = 1
    # Reserve distinct anchors so two nearby objects cannot overwrite each other.
    available = overlaps.clone()
    for target in range(len(boxes)):
        anchor_index = available[:, target].argmax()
        matched[anchor_index] = target
        labels[anchor_index] = 1
        available[anchor_index, :] = -1
    positive = labels == 1
    offsets[positive] = encode(boxes[matched[positive]], anchors[positive])
    return labels, offsets


def detection_loss(logits, predictions, anchors, targets):
    classification = logits.sum() * 0
    localization = predictions.sum() * 0
    total_positive = 0
    for image_logits, image_offsets, boxes in zip(logits, predictions, targets):
        labels, expected = match(anchors, boxes)
        positive = labels == 1
        count = int(positive.sum())
        total_positive += count
        losses = functional.cross_entropy(image_logits, labels, ignore_index=-1, reduction="none")
        negatives = losses.masked_fill(labels != 0, -float("inf"))
        negative_count = min(max(3 * count, 32), int((labels == 0).sum()))
        negative_indices = negatives.topk(negative_count).indices
        classification = classification + losses[positive].sum() + losses[negative_indices].sum()
        localization = localization + functional.smooth_l1_loss(
            image_offsets[positive], expected[positive], reduction="sum")
    normalizer = max(total_positive, 1)
    return (classification + localization) / normalizer


def nms(boxes, scores, threshold=0.45, limit=100):
    order = scores.argsort(descending=True, stable=True)
    kept = []
    while len(order) and len(kept) < limit:
        chosen = int(order[0])
        kept.append(chosen)
        overlaps = box_iou(boxes[chosen:chosen + 1], boxes[order[1:]])[0]
        order = order[1:][overlaps <= threshold]
    return torch.tensor(kept, dtype=torch.long)


def detections(logits, offsets, anchors, score_threshold=0.01, nms_threshold=0.45):
    scores = logits.softmax(dim=-1)[:, 1]
    boxes = decode(offsets, anchors).clamp(0, 1)
    valid = (scores >= score_threshold) & ((boxes[:, 2:] - boxes[:, :2]).min(dim=1).values > 0)
    scores, boxes = scores[valid], boxes[valid]
    # Limit work before the deliberately simple CPU NMS loop.
    order = scores.argsort(descending=True, stable=True)[:500]
    boxes, scores = boxes[order], scores[order]
    kept = nms(boxes, scores, nms_threshold)
    return boxes[kept], scores[kept]